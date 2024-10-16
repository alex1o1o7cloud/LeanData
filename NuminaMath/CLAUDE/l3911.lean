import Mathlib

namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3911_391147

open Set

noncomputable def A : Set ℝ := {x | x ≥ -1}
noncomputable def B : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}

theorem intersection_complement_equality : A ∩ (univ \ B) = Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3911_391147


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l3911_391102

theorem gold_coin_distribution (x y : ℤ) (h1 : x - y = 1) (h2 : x^2 - y^2 = 25*(x - y)) : x + y = 25 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l3911_391102


namespace NUMINAMATH_CALUDE_nearest_integer_to_sum_l3911_391123

theorem nearest_integer_to_sum : ∃ (n : ℤ), n = 3 ∧ 
  ∀ (m : ℤ), abs (m - (2007 / 2999 + 8001 / 5998 + 2001 / 3999 : ℚ)) ≥ 
              abs (n - (2007 / 2999 + 8001 / 5998 + 2001 / 3999 : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_nearest_integer_to_sum_l3911_391123


namespace NUMINAMATH_CALUDE_chubby_checkerboard_black_squares_l3911_391136

/-- Represents a square on the checkerboard -/
inductive Square
| Black
| Red

/-- Represents a checkerboard -/
def Checkerboard := Array (Array Square)

/-- Creates a checkerboard with the given dimensions and pattern -/
def createCheckerboard (n : Nat) : Checkerboard :=
  sorry

/-- Counts the number of black squares on the checkerboard -/
def countBlackSquares (board : Checkerboard) : Nat :=
  sorry

theorem chubby_checkerboard_black_squares :
  let board := createCheckerboard 29
  countBlackSquares board = 421 := by
  sorry

end NUMINAMATH_CALUDE_chubby_checkerboard_black_squares_l3911_391136


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l3911_391132

theorem sin_cos_pi_12 : 2 * Real.sin (π / 12) * Real.cos (π / 12) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l3911_391132


namespace NUMINAMATH_CALUDE_seeds_per_can_l3911_391118

def total_seeds : ℝ := 54.0
def num_cans : ℝ := 9.0

theorem seeds_per_can :
  total_seeds / num_cans = 6.0 :=
by sorry

end NUMINAMATH_CALUDE_seeds_per_can_l3911_391118


namespace NUMINAMATH_CALUDE_sum_sqrt_inequality_l3911_391188

theorem sum_sqrt_inequality (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_sqrt_inequality_l3911_391188


namespace NUMINAMATH_CALUDE_fruit_cost_l3911_391173

/-- The cost of fruits for Michael --/
theorem fruit_cost (a b c d : ℚ) : 
  a + b + c + d = 33 →
  d = 3 * a →
  c = a + 2 * b →
  b + c = 13 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_l3911_391173


namespace NUMINAMATH_CALUDE_distance_B_to_center_l3911_391116

/-- A circle with radius √52 and points A, B, C satisfying given conditions -/
structure NotchedCircle where
  -- Define the circle
  center : ℝ × ℝ
  radius : ℝ
  radius_eq : radius = Real.sqrt 52

  -- Define points A, B, C
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

  -- Conditions
  on_circle_A : (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2
  on_circle_B : (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2
  on_circle_C : (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2

  AB_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64  -- 8^2 = 64
  BC_length : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 16  -- 4^2 = 16

  right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0

/-- The square of the distance from point B to the center of the circle is 20 -/
theorem distance_B_to_center (nc : NotchedCircle) :
  (nc.B.1 - nc.center.1)^2 + (nc.B.2 - nc.center.2)^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_distance_B_to_center_l3911_391116


namespace NUMINAMATH_CALUDE_number_control_l3911_391101

def increase_number (n : ℕ) : ℕ := n + 102

def can_rearrange_to_three_digits (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ n = a * 100 + b * 10 + c

theorem number_control (start : ℕ) (h_start : start = 123) :
  ∀ (t : ℕ), ∃ (n : ℕ), 
    n ≤ increase_number^[t] start ∧
    can_rearrange_to_three_digits n :=
by sorry

end NUMINAMATH_CALUDE_number_control_l3911_391101


namespace NUMINAMATH_CALUDE_bobs_age_multiple_l3911_391163

theorem bobs_age_multiple (bob_age carol_age : ℕ) (m : ℚ) : 
  bob_age = 16 →
  carol_age = 50 →
  carol_age = m * bob_age + 2 →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_bobs_age_multiple_l3911_391163


namespace NUMINAMATH_CALUDE_mixture_ratio_l3911_391104

/-- Proves that combining 5 liters of Mixture A (2/3 alcohol, 1/3 water) with 14 liters of Mixture B (4/7 alcohol, 3/7 water) results in a mixture with an alcohol to water volume ratio of 34:23 -/
theorem mixture_ratio (mixture_a_volume : ℚ) (mixture_b_volume : ℚ)
  (mixture_a_alcohol_ratio : ℚ) (mixture_a_water_ratio : ℚ)
  (mixture_b_alcohol_ratio : ℚ) (mixture_b_water_ratio : ℚ)
  (h1 : mixture_a_volume = 5)
  (h2 : mixture_b_volume = 14)
  (h3 : mixture_a_alcohol_ratio = 2/3)
  (h4 : mixture_a_water_ratio = 1/3)
  (h5 : mixture_b_alcohol_ratio = 4/7)
  (h6 : mixture_b_water_ratio = 3/7) :
  (mixture_a_volume * mixture_a_alcohol_ratio + mixture_b_volume * mixture_b_alcohol_ratio) /
  (mixture_a_volume * mixture_a_water_ratio + mixture_b_volume * mixture_b_water_ratio) = 34/23 :=
by sorry

end NUMINAMATH_CALUDE_mixture_ratio_l3911_391104


namespace NUMINAMATH_CALUDE_einstein_born_on_friday_l3911_391175

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Checks if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  year % 400 == 0 || (year % 4 == 0 && year % 100 ≠ 0)

/-- Einstein's birth year -/
def einsteinBirthYear : Nat := 1865

/-- Einstein's 160th anniversary year -/
def anniversaryYear : Nat := 2025

/-- Day of the week of Einstein's 160th anniversary -/
def anniversaryDayOfWeek : DayOfWeek := DayOfWeek.Friday

/-- Calculates the day of the week Einstein was born -/
def einsteinBirthDayOfWeek : DayOfWeek := sorry

theorem einstein_born_on_friday :
  einsteinBirthDayOfWeek = DayOfWeek.Friday := by sorry

end NUMINAMATH_CALUDE_einstein_born_on_friday_l3911_391175


namespace NUMINAMATH_CALUDE_triangle_side_length_l3911_391191

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  b = 6 ∧
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 ∧
  A = π/3 →
  a = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3911_391191


namespace NUMINAMATH_CALUDE_largest_x_value_l3911_391196

theorem largest_x_value (x : ℝ) : 
  (x / 4 + 1 / (6 * x) = 2 / 3) → 
  x ≤ (4 + Real.sqrt 10) / 3 ∧ 
  ∃ y, y / 4 + 1 / (6 * y) = 2 / 3 ∧ y = (4 + Real.sqrt 10) / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l3911_391196


namespace NUMINAMATH_CALUDE_complex_magnitude_l3911_391117

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - 2 * w) = 15)
  (h2 : Complex.abs (2 * z + 3 * w) = 10)
  (h3 : Complex.abs (z - w) = 3) :
  Complex.abs z = 4.5 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3911_391117


namespace NUMINAMATH_CALUDE_inequality_range_l3911_391122

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x - 1| > a^2 - 2*a) → 
  -1 < a ∧ a < 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l3911_391122


namespace NUMINAMATH_CALUDE_gravel_cost_is_correct_l3911_391174

/-- Represents the dimensions and cost parameters of a rectangular plot with a gravel path --/
structure PlotWithPath where
  length : Real
  width : Real
  pathWidth : Real
  gravelCost : Real

/-- Calculates the cost of gravelling the path for a given plot --/
def calculateGravellingCost (plot : PlotWithPath) : Real :=
  let outerArea := plot.length * plot.width
  let innerLength := plot.length - 2 * plot.pathWidth
  let innerWidth := plot.width - 2 * plot.pathWidth
  let innerArea := innerLength * innerWidth
  let pathArea := outerArea - innerArea
  pathArea * plot.gravelCost

/-- Theorem stating that the cost of gravelling the path for the given plot is 8.844 rupees --/
theorem gravel_cost_is_correct (plot : PlotWithPath) 
  (h1 : plot.length = 110)
  (h2 : plot.width = 0.65)
  (h3 : plot.pathWidth = 0.05)
  (h4 : plot.gravelCost = 0.8) :
  calculateGravellingCost plot = 8.844 := by
  sorry

#eval calculateGravellingCost { length := 110, width := 0.65, pathWidth := 0.05, gravelCost := 0.8 }

end NUMINAMATH_CALUDE_gravel_cost_is_correct_l3911_391174


namespace NUMINAMATH_CALUDE_jaime_sum_with_square_l3911_391121

theorem jaime_sum_with_square (n : ℕ) (k : ℕ) : 
  (∃ (i : ℕ), i < 100 ∧ n + i = k) →
  (50 * (2 * n + 99) - k + k^2 = 7500) →
  k = 26 := by
sorry

end NUMINAMATH_CALUDE_jaime_sum_with_square_l3911_391121


namespace NUMINAMATH_CALUDE_ellipse_equation_with_given_parameters_l3911_391152

/-- Standard equation of an ellipse with foci on coordinate axes -/
def standard_ellipse_equation (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 / a^2 + y^2 / b^2 = 1}

/-- Theorem: Standard equation of ellipse with given parameters -/
theorem ellipse_equation_with_given_parameters :
  ∀ (a c : ℝ),
  a^2 = 13 →
  c^2 = 12 →
  ∃ (b : ℝ),
  b^2 = 1 ∧
  (standard_ellipse_equation 13 1 = standard_ellipse_equation 13 1 ∨
   standard_ellipse_equation 1 13 = standard_ellipse_equation 1 13) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_with_given_parameters_l3911_391152


namespace NUMINAMATH_CALUDE_junior_score_l3911_391154

theorem junior_score (n : ℝ) (junior_ratio : ℝ) (senior_ratio : ℝ) 
  (overall_avg : ℝ) (senior_avg : ℝ) (h1 : junior_ratio = 0.2) 
  (h2 : senior_ratio = 0.8) (h3 : junior_ratio + senior_ratio = 1) 
  (h4 : overall_avg = 85) (h5 : senior_avg = 84) : 
  (overall_avg * n - senior_avg * senior_ratio * n) / (junior_ratio * n) = 89 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l3911_391154


namespace NUMINAMATH_CALUDE_wall_penetrating_skill_l3911_391170

theorem wall_penetrating_skill (n : ℕ) : 
  (8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n))) ↔ n = 63 := by
  sorry

end NUMINAMATH_CALUDE_wall_penetrating_skill_l3911_391170


namespace NUMINAMATH_CALUDE_rectangle_square_diagonal_intersection_l3911_391139

/-- Given a square and a rectangle with the same perimeter and a common corner,
    prove that the intersection of the rectangle's diagonals lies on the square's diagonal. -/
theorem rectangle_square_diagonal_intersection
  (s a b : ℝ) 
  (h_perimeter : 4 * s = 2 * a + 2 * b) 
  (h_positive : s > 0 ∧ a > 0 ∧ b > 0) :
  a / 2 = b / 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_square_diagonal_intersection_l3911_391139


namespace NUMINAMATH_CALUDE_ferry_tourists_sum_l3911_391164

/-- The number of trips the ferry makes in a day -/
def num_trips : ℕ := 6

/-- The number of tourists on the first trip -/
def initial_tourists : ℕ := 100

/-- The decrease in number of tourists per trip -/
def tourist_decrease : ℕ := 1

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The total number of tourists transported by the ferry in a day -/
def total_tourists : ℕ := arithmetic_sum initial_tourists tourist_decrease num_trips

theorem ferry_tourists_sum :
  total_tourists = 585 := by sorry

end NUMINAMATH_CALUDE_ferry_tourists_sum_l3911_391164


namespace NUMINAMATH_CALUDE_equation_solution_l3911_391169

theorem equation_solution :
  ∀ x : ℝ, (Real.sqrt (9 * x - 2) + 18 / Real.sqrt (9 * x - 2) = 11) ↔ (x = 83 / 9 ∨ x = 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3911_391169


namespace NUMINAMATH_CALUDE_stock_quoted_value_l3911_391185

/-- Proves that given an investment of 1620 in an 8% stock that earns 135, the stock is quoted at 96 --/
theorem stock_quoted_value (investment : ℝ) (dividend_rate : ℝ) (dividend_earned : ℝ) 
  (h1 : investment = 1620)
  (h2 : dividend_rate = 8 / 100)
  (h3 : dividend_earned = 135) :
  (investment / ((dividend_earned * 100) / dividend_rate)) * 100 = 96 := by
  sorry

end NUMINAMATH_CALUDE_stock_quoted_value_l3911_391185


namespace NUMINAMATH_CALUDE_constant_dot_product_l3911_391140

/-- The ellipse E -/
def ellipse_E (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- The hyperbola C -/
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/2 = 1

/-- The foci of ellipse E and hyperbola C coincide -/
axiom foci_coincide : ∀ x y : ℝ, ellipse_E x y → hyperbola_C x y → x^2 - y^2 = 3

/-- The minor axis endpoints and one focus of ellipse E form an equilateral triangle -/
axiom equilateral_triangle : ∀ x y : ℝ, ellipse_E x y → x^2 + y^2 = 1 → x^2 = 3/4

/-- The dot product MP · MQ is constant when m = 17/8 -/
theorem constant_dot_product :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  ellipse_E x₁ y₁ → ellipse_E x₂ y₂ →
  ∃ k : ℝ, y₁ = k * (x₁ - 1) ∧ y₂ = k * (x₂ - 1) →
  (17/8 - x₁) * (17/8 - x₂) + y₁ * y₂ = 33/64 :=
sorry

end NUMINAMATH_CALUDE_constant_dot_product_l3911_391140


namespace NUMINAMATH_CALUDE_unwashed_shirts_l3911_391142

theorem unwashed_shirts 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 21)
  (h3 : washed = 29) : 
  short_sleeve + long_sleeve - washed = 1 := by
  sorry

end NUMINAMATH_CALUDE_unwashed_shirts_l3911_391142


namespace NUMINAMATH_CALUDE_geometric_sum_n1_l3911_391133

theorem geometric_sum_n1 (a : ℝ) (h : a ≠ 1) :
  1 + a + a^2 = (1 - a^3) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_n1_l3911_391133


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3911_391190

theorem fraction_evaluation : 
  (20-19+18-17+16-15+14-13+12-11+10-9+8-7+6-5+4-3+2-1) / 
  (2-3+4-5+6-7+8-9+10-11+12-13+14-15+16-17+18-19+20) = 10/11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3911_391190


namespace NUMINAMATH_CALUDE_sam_average_letters_per_day_l3911_391124

/-- The average number of letters Sam wrote per day -/
theorem sam_average_letters_per_day :
  let tuesday_letters : ℕ := 7
  let wednesday_letters : ℕ := 3
  let total_days : ℕ := 2
  let total_letters : ℕ := tuesday_letters + wednesday_letters
  (total_letters : ℚ) / total_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_sam_average_letters_per_day_l3911_391124


namespace NUMINAMATH_CALUDE_sin_shift_equivalence_l3911_391192

theorem sin_shift_equivalence :
  ∀ x : ℝ, Real.sin (2 * x + π / 3) = Real.sin (2 * (x + π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_equivalence_l3911_391192


namespace NUMINAMATH_CALUDE_casino_solution_l3911_391135

def casino_problem (money_A money_B money_C : ℕ) : Prop :=
  (money_B = 2 * money_C) ∧
  (money_A = 40) ∧
  (money_A + money_B + money_C = 220)

theorem casino_solution :
  ∀ money_A money_B money_C,
    casino_problem money_A money_B money_C →
    money_C - money_A = 20 := by
  sorry

end NUMINAMATH_CALUDE_casino_solution_l3911_391135


namespace NUMINAMATH_CALUDE_walking_time_proportional_l3911_391110

/-- Given a constant walking rate, prove that if it takes 6 minutes to walk 2 miles, 
    then it will take 12 minutes to walk 4 miles. -/
theorem walking_time_proportional (rate : ℝ) : 
  (rate * 2 = 6) → (rate * 4 = 12) := by
  sorry

end NUMINAMATH_CALUDE_walking_time_proportional_l3911_391110


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3911_391166

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 141 →
  divisor = 17 →
  quotient = 8 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3911_391166


namespace NUMINAMATH_CALUDE_book_arrangements_eq_120960_l3911_391153

/-- The number of ways to arrange 4 different math books and 5 different history books on a bookshelf,
    with a math book at both ends and exactly one math book in the middle -/
def book_arrangements : ℕ :=
  let math_books := 4
  let history_books := 5
  let end_arrangements := math_books * (math_books - 1)
  let middle_math_book := math_books - 2
  let remaining_books := (math_books - 3) + history_books
  end_arrangements * middle_math_book * Nat.factorial remaining_books

theorem book_arrangements_eq_120960 : book_arrangements = 120960 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangements_eq_120960_l3911_391153


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3911_391167

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a with a 0 = 3, a 1 = 7, a n = x,
    a (n+1) = y, a (n+2) = t, a (n+3) = 35, and t = 31,
    prove that x + y = 50 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 0 = 3)
  (h3 : a 1 = 7)
  (h4 : a n = x)
  (h5 : a (n+1) = y)
  (h6 : a (n+2) = t)
  (h7 : a (n+3) = 35)
  (h8 : t = 31) :
  x + y = 50 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3911_391167


namespace NUMINAMATH_CALUDE_remove_four_gives_desired_average_l3911_391127

def original_list : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def remove_number (list : List Nat) (n : Nat) : List Nat :=
  list.filter (· ≠ n)

def average (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

theorem remove_four_gives_desired_average :
  average (remove_number original_list 4) = 29/4 := by
sorry

end NUMINAMATH_CALUDE_remove_four_gives_desired_average_l3911_391127


namespace NUMINAMATH_CALUDE_not_p_neither_sufficient_nor_necessary_for_q_l3911_391184

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

-- Define what it means for ¬p to be neither sufficient nor necessary for q
def neither_sufficient_nor_necessary : Prop :=
  (∃ x : ℝ, ¬(p x) ∧ ¬(q x)) ∧ 
  (∃ y : ℝ, ¬(p y) ∧ q y) ∧ 
  (∃ z : ℝ, p z ∧ q z)

-- Theorem statement
theorem not_p_neither_sufficient_nor_necessary_for_q : 
  neither_sufficient_nor_necessary :=
sorry

end NUMINAMATH_CALUDE_not_p_neither_sufficient_nor_necessary_for_q_l3911_391184


namespace NUMINAMATH_CALUDE_share_difference_l3911_391177

theorem share_difference (total : ℕ) (a b c : ℕ) : 
  total = 120 →
  a = b + 20 →
  a < c →
  b = 20 →
  c - a = 20 := by
sorry

end NUMINAMATH_CALUDE_share_difference_l3911_391177


namespace NUMINAMATH_CALUDE_function_value_when_previous_is_one_l3911_391189

theorem function_value_when_previous_is_one 
  (f : ℤ → ℤ) 
  (h1 : ∀ n : ℤ, f n = f (n - 1) - n) 
  (h2 : f 4 = 12) :
  ∀ n : ℤ, f (n - 1) = 1 → f n = 7 := by
sorry

end NUMINAMATH_CALUDE_function_value_when_previous_is_one_l3911_391189


namespace NUMINAMATH_CALUDE_cavalier_projection_triangle_area_l3911_391179

/-- Given a right-angled triangle represented in an oblique cavalier projection
    with a hypotenuse of √2a, prove that its area is √2a² -/
theorem cavalier_projection_triangle_area (a : ℝ) (h : a > 0) :
  let leg1 := Real.sqrt 2 * a
  let leg2 := 2 * a
  (1 / 2) * leg1 * leg2 = Real.sqrt 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_cavalier_projection_triangle_area_l3911_391179


namespace NUMINAMATH_CALUDE_problem_statement_l3911_391146

theorem problem_statement (a : ℝ) (h1 : a > 1) (h2 : 20 * a / (a^2 + 1) = Real.sqrt 2) :
  14 * a / (a^2 - 1) = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3911_391146


namespace NUMINAMATH_CALUDE_only_one_divisible_l3911_391144

theorem only_one_divisible (n : ℕ+) : (3^(n : ℕ) + 1) % (n : ℕ)^2 = 0 → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_one_divisible_l3911_391144


namespace NUMINAMATH_CALUDE_sqrt_simplification_l3911_391129

theorem sqrt_simplification : (5 - 3 * Real.sqrt 2) ^ 2 = 45 - 28 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l3911_391129


namespace NUMINAMATH_CALUDE_tangent_line_of_odd_function_l3911_391182

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a - 2) * x^2 + 2 * x

-- State the theorem
theorem tangent_line_of_odd_function (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) →  -- f is an odd function
  (f a 1 = 3) →                 -- f(1) = 3
  ∃ m b : ℝ, m = 5 ∧ b = -2 ∧
    ∀ x, (f a x - f a 1) = m * (x - 1) + b :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_of_odd_function_l3911_391182


namespace NUMINAMATH_CALUDE_right_triangle_area_l3911_391193

theorem right_triangle_area (p : ℝ) (h : p > 0) : ∃ (x y z : ℝ),
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x^2 + y^2 = z^2 ∧
  x + y + z = 3*p ∧
  x = z/2 ∧
  (1/2) * x * y = (p^2 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3911_391193


namespace NUMINAMATH_CALUDE_marias_green_towels_l3911_391161

theorem marias_green_towels :
  ∀ (green_towels : ℕ),
  (green_towels + 21 : ℕ) - 34 = 22 →
  green_towels = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_marias_green_towels_l3911_391161


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l3911_391172

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 ∧ m < n → (¬(24 ∣ m^2) ∨ ¬(720 ∣ m^3))) ∧ 
  24 ∣ n^2 ∧ 720 ∣ n^3 ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l3911_391172


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3911_391115

theorem diophantine_equation_solutions :
  ∀ (x y : ℕ) (p : ℕ), 
    Prime p → 
    p^x - y^p = 1 → 
    ((x = 1 ∧ y = 1 ∧ p = 2) ∨ (x = 2 ∧ y = 2 ∧ p = 3)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3911_391115


namespace NUMINAMATH_CALUDE_tables_needed_l3911_391119

theorem tables_needed (invited : ℕ) (no_shows : ℕ) (capacity : ℕ) : 
  invited = 24 → no_shows = 10 → capacity = 7 → 
  (invited - no_shows + capacity - 1) / capacity = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_tables_needed_l3911_391119


namespace NUMINAMATH_CALUDE_last_digit_89_base5_l3911_391113

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem last_digit_89_base5 : 
  (decimal_to_base5 89).getLast? = some 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_89_base5_l3911_391113


namespace NUMINAMATH_CALUDE_fishing_catch_difference_l3911_391194

theorem fishing_catch_difference (father_catch son_catch transfer : ℚ) : 
  (father_catch - transfer = son_catch + transfer) →
  (father_catch + transfer = 2 * (son_catch - transfer)) →
  (father_catch - son_catch) / son_catch = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_fishing_catch_difference_l3911_391194


namespace NUMINAMATH_CALUDE_chebyshev_properties_l3911_391160

/-- Chebyshev polynomial of the first kind -/
def T : ℕ → (Real → Real)
| 0 => λ _ => 1
| 1 => λ x => x
| (n + 2) => λ x => 2 * x * T (n + 1) x - T n x

/-- Chebyshev polynomial of the second kind -/
def U : ℕ → (Real → Real)
| 0 => λ _ => 1
| 1 => λ x => 2 * x
| (n + 2) => λ x => 2 * x * U (n + 1) x - U n x

/-- Theorem: Chebyshev polynomials satisfy their initial conditions and recurrence relations -/
theorem chebyshev_properties :
  (∀ x, T 0 x = 1) ∧
  (∀ x, T 1 x = x) ∧
  (∀ n x, T (n + 1) x = 2 * x * T n x - T (n - 1) x) ∧
  (∀ x, U 0 x = 1) ∧
  (∀ x, U 1 x = 2 * x) ∧
  (∀ n x, U (n + 1) x = 2 * x * U n x - U (n - 1) x) := by
  sorry

end NUMINAMATH_CALUDE_chebyshev_properties_l3911_391160


namespace NUMINAMATH_CALUDE_every_second_sum_of_arithmetic_sequence_l3911_391134

def sequence_sum (first : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * first + (n - 1)) / 2

def every_second_sum (first : ℚ) (n : ℕ) : ℚ :=
  sequence_sum first ((n + 1) / 2)

theorem every_second_sum_of_arithmetic_sequence 
  (first : ℚ) (n : ℕ) (h1 : n = 3015) (h2 : sequence_sum first n = 8010) :
  every_second_sum first (n - 1) = 3251.5 := by
  sorry

end NUMINAMATH_CALUDE_every_second_sum_of_arithmetic_sequence_l3911_391134


namespace NUMINAMATH_CALUDE_fraction_of_72_l3911_391148

theorem fraction_of_72 : (1 / 3 : ℚ) * (1 / 4 : ℚ) * (1 / 6 : ℚ) * 72 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_72_l3911_391148


namespace NUMINAMATH_CALUDE_determine_new_harvest_l3911_391157

/-- Represents the harvest data for two plots of land before and after applying new agricultural techniques. -/
structure HarvestData where
  initial_total : ℝ
  yield_increase_plot1 : ℝ
  yield_increase_plot2 : ℝ
  new_total : ℝ

/-- Represents the harvest amounts for each plot after applying new techniques. -/
structure NewHarvest where
  plot1 : ℝ
  plot2 : ℝ

/-- Theorem stating that given the initial conditions, the new harvest amounts can be determined. -/
theorem determine_new_harvest (data : HarvestData) 
  (h1 : data.initial_total = 14.7)
  (h2 : data.yield_increase_plot1 = 0.8)
  (h3 : data.yield_increase_plot2 = 0.24)
  (h4 : data.new_total = 21.42) :
  ∃ (new_harvest : NewHarvest),
    new_harvest.plot1 = 10.26 ∧
    new_harvest.plot2 = 11.16 ∧
    new_harvest.plot1 + new_harvest.plot2 = data.new_total ∧
    new_harvest.plot1 / (1 + data.yield_increase_plot1) + 
    new_harvest.plot2 / (1 + data.yield_increase_plot2) = data.initial_total :=
  sorry

end NUMINAMATH_CALUDE_determine_new_harvest_l3911_391157


namespace NUMINAMATH_CALUDE_function_through_points_l3911_391171

theorem function_through_points (a p q : ℝ) : 
  a > 0 →
  2^p / (2^p + a*p) = 6/5 →
  2^q / (2^q + a*q) = -1/5 →
  2^(p+q) = 16*p*q →
  a = 4 :=
by sorry

end NUMINAMATH_CALUDE_function_through_points_l3911_391171


namespace NUMINAMATH_CALUDE_triangle_theorem_triangle_area_l3911_391181

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi)
  (h2 : Real.sin t.B * (Real.tan t.A + Real.tan t.C) = Real.tan t.A * Real.tan t.C) :
  t.b^2 = t.a * t.c :=
sorry

/-- The area theorem when a = 2c = 2 -/
theorem triangle_area (t : Triangle)
  (h1 : t.A + t.B + t.C = Real.pi)
  (h2 : Real.sin t.B * (Real.tan t.A + Real.tan t.C) = Real.tan t.A * Real.tan t.C)
  (h3 : t.a = 2 * t.c)
  (h4 : t.a = 2) :
  (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 7 / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_triangle_area_l3911_391181


namespace NUMINAMATH_CALUDE_quadratic_roots_l3911_391100

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 0 ∧ x₂ = 4/5) ∧ 
  (∀ x : ℝ, 5 * x^2 = 4 * x ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3911_391100


namespace NUMINAMATH_CALUDE_square_area_ratio_l3911_391162

theorem square_area_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := s₂ * Real.sqrt 2
  (s₁ ^ 2) / (s₂ ^ 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3911_391162


namespace NUMINAMATH_CALUDE_soccer_season_games_l3911_391199

/-- Represents a soccer team's season performance -/
structure SoccerSeason where
  totalGames : ℕ
  firstGames : ℕ
  firstWins : ℕ
  remainingWins : ℕ

/-- Conditions for the soccer season -/
def validSeason (s : SoccerSeason) : Prop :=
  s.totalGames % 2 = 0 ∧ 
  s.firstGames = 36 ∧
  s.firstWins = 16 ∧
  s.remainingWins ≥ (s.totalGames - s.firstGames) * 3 / 4 ∧
  (s.firstWins + s.remainingWins) * 100 = s.totalGames * 62

theorem soccer_season_games (s : SoccerSeason) (h : validSeason s) : s.totalGames = 84 :=
sorry

end NUMINAMATH_CALUDE_soccer_season_games_l3911_391199


namespace NUMINAMATH_CALUDE_middle_zero_product_l3911_391183

theorem middle_zero_product (a b c d : ℕ) : ∃ (x y z w : ℕ), 
  (x ≠ 0 ∧ z ≠ 0 ∧ y ≠ 0 ∧ w ≠ 0) ∧ 
  (100 * x + 0 * 10 + y) * z = 100 * a + 0 * 10 + b ∧
  (100 * x + 0 * 10 + y) * w = 100 * c + d * 10 + e ∧
  d ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_middle_zero_product_l3911_391183


namespace NUMINAMATH_CALUDE_sin_cos_extrema_l3911_391168

theorem sin_cos_extrema (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  let m := Real.sin x - Real.cos y ^ 2
  (∀ a b, Real.sin a + Real.sin b = 1/3 → m ≤ Real.sin a - Real.cos b ^ 2) ∧
  (∃ x' y', Real.sin x' + Real.sin y' = 1/3 ∧ m = 4/9) ∧
  (∀ a b, Real.sin a + Real.sin b = 1/3 → Real.sin a - Real.cos b ^ 2 ≤ m) ∧
  (∃ x'' y'', Real.sin x'' + Real.sin y'' = 1/3 ∧ m = -11/16) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_extrema_l3911_391168


namespace NUMINAMATH_CALUDE_total_spent_is_157_l3911_391180

-- Define the initial amount given to each person
def initial_amount : ℕ := 250

-- Define Pete's spending
def pete_spending : ℕ := 20 + 30 + 50 + 5

-- Define Raymond's remaining money
def raymond_remaining : ℕ := 70 + 100 + 25 + 3

-- Theorem to prove
theorem total_spent_is_157 : 
  pete_spending + (initial_amount - raymond_remaining) = 157 := by
  sorry


end NUMINAMATH_CALUDE_total_spent_is_157_l3911_391180


namespace NUMINAMATH_CALUDE_geometry_theorem_l3911_391137

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (intersects : Plane → Plane → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)
variable (perp_lines : Line → Line → Prop)

-- State the theorem
theorem geometry_theorem 
  (α β γ : Plane) (l m : Line)
  (h1 : intersects β γ l)
  (h2 : parallel l α)
  (h3 : contains α m)
  (h4 : perpendicular m γ) :
  perp_planes α γ ∧ perp_lines l m :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l3911_391137


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3911_391197

/-- An arithmetic sequence with sum of first n terms S_n = 2n^2 - 25n -/
def S (n : ℕ) : ℤ := 2 * n^2 - 25 * n

/-- The nth term of the arithmetic sequence -/
def a (n : ℕ) : ℤ := 4 * n - 27

theorem arithmetic_sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n = 4 * n - 27) ∧
  (∀ n : ℕ, n ≠ 6 → S n > S 6) ∧
  S 6 = -78 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3911_391197


namespace NUMINAMATH_CALUDE_longest_to_shortest_l3911_391105

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ
  hyp_short : shorterLeg = hypotenuse / 2
  hyp_long : longerLeg = shorterLeg * Real.sqrt 3

/-- Represents a sequence of four 30-60-90 triangles -/
structure FourTriangles where
  t1 : Triangle30_60_90
  t2 : Triangle30_60_90
  t3 : Triangle30_60_90
  t4 : Triangle30_60_90
  hyp_relation1 : t1.longerLeg = t2.hypotenuse
  hyp_relation2 : t2.longerLeg = t3.hypotenuse
  hyp_relation3 : t3.longerLeg = t4.hypotenuse

theorem longest_to_shortest (triangles : FourTriangles) 
    (h : triangles.t1.hypotenuse = 16) : 
    triangles.t4.longerLeg = 9 := by
  sorry

end NUMINAMATH_CALUDE_longest_to_shortest_l3911_391105


namespace NUMINAMATH_CALUDE_marys_cake_recipe_l3911_391187

/-- Mary's cake recipe problem -/
theorem marys_cake_recipe 
  (total_flour : ℕ) 
  (sugar : ℕ) 
  (flour_to_add : ℕ) 
  (h1 : total_flour = 9)
  (h2 : flour_to_add = sugar + 1)
  (h3 : sugar = 6) :
  total_flour - flour_to_add = 2 := by
  sorry

end NUMINAMATH_CALUDE_marys_cake_recipe_l3911_391187


namespace NUMINAMATH_CALUDE_complex_power_difference_l3911_391128

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l3911_391128


namespace NUMINAMATH_CALUDE_inequality_proof_l3911_391131

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ((x + y + z) / 3) ^ (x + y + z) ≤ x^x * y^y * z^z ∧
  x^x * y^y * z^z ≤ ((x^2 + y^2 + z^2) / (x + y + z)) ^ (x + y + z) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3911_391131


namespace NUMINAMATH_CALUDE_current_rate_calculation_l3911_391195

/-- Given a boat traveling downstream, calculate the rate of the current. -/
theorem current_rate_calculation 
  (boat_speed : ℝ)             -- Speed of the boat in still water (km/hr)
  (distance : ℝ)               -- Distance traveled downstream (km)
  (time : ℝ)                   -- Time taken for the downstream journey (hr)
  (h1 : boat_speed = 20)       -- The boat's speed in still water is 20 km/hr
  (h2 : distance = 10)         -- The distance traveled downstream is 10 km
  (h3 : time = 24 / 60)        -- The time taken is 24 minutes, converted to hours
  : ∃ (current_rate : ℝ), 
    distance = (boat_speed + current_rate) * time ∧ 
    current_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_current_rate_calculation_l3911_391195


namespace NUMINAMATH_CALUDE_min_selling_price_A_is_190_l3911_391130

/-- Represents the number of units and prices of water purifiers --/
structure WaterPurifiers where
  units_A : ℕ
  units_B : ℕ
  cost_A : ℕ
  cost_B : ℕ
  total_cost : ℕ

/-- Calculates the minimum selling price of model A --/
def min_selling_price_A (w : WaterPurifiers) : ℕ :=
  w.cost_A + (w.total_cost - w.units_A * w.cost_A - w.units_B * w.cost_B) / w.units_A

/-- Theorem stating the minimum selling price of model A --/
theorem min_selling_price_A_is_190 (w : WaterPurifiers) 
  (h1 : w.units_A + w.units_B = 100)
  (h2 : w.units_A * w.cost_A + w.units_B * w.cost_B = w.total_cost)
  (h3 : w.cost_A = 150)
  (h4 : w.cost_B = 250)
  (h5 : w.total_cost = 19000)
  (h6 : ∀ (sell_A : ℕ), 
    (sell_A - w.cost_A) * w.units_A + 2 * (sell_A - w.cost_A) * w.units_B ≥ 5600 → 
    min_selling_price_A w ≤ sell_A) :
  min_selling_price_A w = 190 := by
  sorry

#eval min_selling_price_A ⟨60, 40, 150, 250, 19000⟩

end NUMINAMATH_CALUDE_min_selling_price_A_is_190_l3911_391130


namespace NUMINAMATH_CALUDE_derivative_at_one_l3911_391111

-- Define the function f(x) = (x+1)^2(x-1)
def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3911_391111


namespace NUMINAMATH_CALUDE_panda_bamboo_transport_l3911_391198

/-- Represents the maximum number of bamboo sticks that can be transported -/
def max_bamboo_transported (initial_bamboo : ℕ) (capacity : ℕ) (consumption : ℕ) : ℕ :=
  initial_bamboo - consumption * (2 * (initial_bamboo / capacity) - 1)

/-- Theorem stating that the maximum number of bamboo sticks transported is 165 -/
theorem panda_bamboo_transport :
  max_bamboo_transported 200 50 5 = 165 := by
  sorry

end NUMINAMATH_CALUDE_panda_bamboo_transport_l3911_391198


namespace NUMINAMATH_CALUDE_exists_increasing_arithmetic_seq_exists_perm_without_long_increasing_seq_l3911_391156

-- Define the set of natural numbers (positive integers)
def N : Set Nat := {n : Nat | n > 0}

-- Define a permutation of N
def isPerm (f : Nat → Nat) : Prop := Function.Bijective f ∧ ∀ n, f n ∈ N

-- Theorem 1
theorem exists_increasing_arithmetic_seq (f : Nat → Nat) (h : isPerm f) :
  ∃ a d : Nat, d > 0 ∧ a ∈ N ∧ (a + d) ∈ N ∧ (a + 2*d) ∈ N ∧
    f a < f (a + d) ∧ f (a + d) < f (a + 2*d) := by sorry

-- Theorem 2
theorem exists_perm_without_long_increasing_seq :
  ∃ f : Nat → Nat, isPerm f ∧
    ∀ a d : Nat, d > 0 → a ∈ N →
      ¬(∀ k : Nat, k ≤ 2003 → f (a + k*d) < f (a + (k+1)*d)) := by sorry

end NUMINAMATH_CALUDE_exists_increasing_arithmetic_seq_exists_perm_without_long_increasing_seq_l3911_391156


namespace NUMINAMATH_CALUDE_max_m_condition_l3911_391143

theorem max_m_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ m : ℝ, 4 / a + 1 / b ≥ m / (a + 4 * b)) →
  (∃ m_max : ℝ, ∀ m : ℝ, m ≤ m_max ∧ (m = m_max ↔ b / a = 1 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_max_m_condition_l3911_391143


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3911_391178

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3911_391178


namespace NUMINAMATH_CALUDE_semicircle_covering_l3911_391149

theorem semicircle_covering (N : ℕ) (r : ℝ) : 
  N > 0 → 
  r > 0 → 
  let A := N * (π * r^2 / 2)
  let B := (π * (N * r)^2 / 2) - A
  A / B = 1 / 3 → 
  N = 4 := by
sorry

end NUMINAMATH_CALUDE_semicircle_covering_l3911_391149


namespace NUMINAMATH_CALUDE_king_ace_probability_l3911_391120

/-- Standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Probability of drawing a King first and an Ace second from a standard deck -/
theorem king_ace_probability :
  (NumKings : ℚ) / StandardDeck * NumAces / (StandardDeck - 1) = 4 / 663 := by
sorry

end NUMINAMATH_CALUDE_king_ace_probability_l3911_391120


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3911_391138

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 3 * y' = 1 → 1 / x' + 1 / y' ≥ 1 / x + 1 / y) →
  1 / x + 1 / y = 3 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3911_391138


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3911_391141

theorem inequality_system_solution (x : ℝ) :
  (5 * x - 2 < 3 * (x + 2)) ∧
  ((2 * x - 1) / 3 - (5 * x + 1) / 2 ≤ 1) →
  -1 ≤ x ∧ x < 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3911_391141


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3911_391107

theorem vector_equation_solution (e₁ e₂ : ℝ × ℝ) (x y : ℝ) 
  (h_not_collinear : ¬ ∃ (k : ℝ), e₂ = k • e₁)
  (h_equation : (3*x - 4*y) • e₁ + (2*x - 3*y) • e₂ = 6 • e₁ + 3 • e₂) :
  x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3911_391107


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l3911_391150

theorem difference_of_squares_special_case : (503 : ℤ) * 503 - 502 * 504 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l3911_391150


namespace NUMINAMATH_CALUDE_common_tangent_intersection_l3911_391176

/-- Ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Ellipse C₂ -/
def C₂ (x y : ℝ) : Prop := (x-2)^2 + 4*y^2 = 1

/-- Common tangent to C₁ and C₂ -/
def common_tangent (x y : ℝ) : Prop :=
  ∃ (k b : ℝ), y = k*x + b ∧
    (∀ x' y', C₁ x' y' → (y' - (k*x' + b))^2 ≥ (k*(x - x'))^2) ∧
    (∀ x' y', C₂ x' y' → (y' - (k*x' + b))^2 ≥ (k*(x - x'))^2)

theorem common_tangent_intersection :
  ∃ (x y : ℝ), common_tangent x y ∧ y = 0 ∧ x = 4 :=
sorry

end NUMINAMATH_CALUDE_common_tangent_intersection_l3911_391176


namespace NUMINAMATH_CALUDE_abc_divides_sum_pow13_l3911_391159

theorem abc_divides_sum_pow13 (a b c : ℕ+) 
  (h1 : a ∣ b^3) 
  (h2 : b ∣ c^3) 
  (h3 : c ∣ a^3) : 
  (a * b * c) ∣ (a + b + c)^13 := by
  sorry

end NUMINAMATH_CALUDE_abc_divides_sum_pow13_l3911_391159


namespace NUMINAMATH_CALUDE_max_rabbits_proof_l3911_391155

/-- Represents the properties of a rabbit -/
structure RabbitProperties :=
  (long_ears : Bool)
  (can_jump_far : Bool)

/-- The maximum number of rabbits satisfying the given conditions -/
def max_rabbits : Nat := 27

/-- Theorem stating that 27 is the maximum number of rabbits satisfying the given conditions -/
theorem max_rabbits_proof :
  ∀ (rabbits : Finset RabbitProperties),
    (rabbits.card ≤ max_rabbits) →
    ((rabbits.filter (λ r => r.long_ears)).card = 13) →
    ((rabbits.filter (λ r => r.can_jump_far)).card = 17) →
    ((rabbits.filter (λ r => r.long_ears ∧ r.can_jump_far)).card ≥ 3) →
    rabbits.card = max_rabbits :=
  sorry

end NUMINAMATH_CALUDE_max_rabbits_proof_l3911_391155


namespace NUMINAMATH_CALUDE_sequence_2024th_term_l3911_391106

/-- Definition of the sequence term -/
def sequenceTerm (n : ℕ) : ℤ × ℕ := ((-1)^(n+1) * (2*n - 1), n)

/-- The 2024th term of the sequence -/
def term2024 : ℤ × ℕ := sequenceTerm 2024

/-- Theorem stating the 2024th term of the sequence -/
theorem sequence_2024th_term :
  term2024 = (-4047, 2024) := by sorry

end NUMINAMATH_CALUDE_sequence_2024th_term_l3911_391106


namespace NUMINAMATH_CALUDE_min_value_sum_ratios_l3911_391108

theorem min_value_sum_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b) + (b / c) + (c / a) + (b / a) ≥ 4 ∧
  ((a / b) + (b / c) + (c / a) + (b / a) = 4 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_ratios_l3911_391108


namespace NUMINAMATH_CALUDE_power_relation_l3911_391112

theorem power_relation (a m n : ℝ) (hm : a^m = 2) (hn : a^n = 5) : 
  a^(3*m - 2*n) = 8/25 := by
sorry

end NUMINAMATH_CALUDE_power_relation_l3911_391112


namespace NUMINAMATH_CALUDE_x_squared_minus_one_is_quadratic_l3911_391165

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x^2 - 1 = 0 is a quadratic equation in one variable -/
theorem x_squared_minus_one_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_one_is_quadratic_l3911_391165


namespace NUMINAMATH_CALUDE_cookie_markup_is_twenty_percent_l3911_391109

/-- The percentage markup on cookies sold by Joe -/
def percentage_markup (num_cookies : ℕ) (total_earned : ℚ) (cost_per_cookie : ℚ) : ℚ :=
  ((total_earned / num_cookies.cast) / cost_per_cookie - 1) * 100

/-- Theorem stating that the percentage markup is 20% given the problem conditions -/
theorem cookie_markup_is_twenty_percent :
  let num_cookies : ℕ := 50
  let total_earned : ℚ := 60
  let cost_per_cookie : ℚ := 1
  percentage_markup num_cookies total_earned cost_per_cookie = 20 := by
sorry

end NUMINAMATH_CALUDE_cookie_markup_is_twenty_percent_l3911_391109


namespace NUMINAMATH_CALUDE_mashas_balls_l3911_391145

theorem mashas_balls (r w n p : ℕ) : 
  r + n * w = 101 →
  p * r + w = 103 →
  (r + w = 51 ∨ r + w = 68) :=
by sorry

end NUMINAMATH_CALUDE_mashas_balls_l3911_391145


namespace NUMINAMATH_CALUDE_equilateral_triangle_height_l3911_391158

/-- Given an equilateral triangle with two vertices at (0,0) and (10,0), 
    and the third vertex (x,y) in the first quadrant, 
    prove that the y-coordinate of the third vertex is 5√3. -/
theorem equilateral_triangle_height : 
  ∀ (x y : ℝ), 
  x ≥ 0 → y > 0 →  -- First quadrant condition
  (x^2 + y^2 = 100) →  -- Distance from (0,0) to (x,y) is 10
  ((x-10)^2 + y^2 = 100) →  -- Distance from (10,0) to (x,y) is 10
  y = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_height_l3911_391158


namespace NUMINAMATH_CALUDE_subtraction_problem_l3911_391103

theorem subtraction_problem : ∃ x : ℕ, x - 56 = 11 ∧ x = 67 := by sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3911_391103


namespace NUMINAMATH_CALUDE_pyramid_frustum_theorem_l3911_391114

-- Define the pyramid
structure Pyramid :=
  (base_side : ℝ)
  (height : ℝ)

-- Define the frustum
structure Frustum :=
  (base_side : ℝ)
  (top_side : ℝ)
  (height : ℝ)

-- Define the theorem
theorem pyramid_frustum_theorem (P : Pyramid) (F : Frustum) (P' : Pyramid) :
  P.base_side = 10 →
  P.height = 15 →
  F.base_side = P.base_side →
  F.top_side = P'.base_side →
  F.height + P'.height = P.height →
  (P.base_side^2 * P.height) = 9 * (P'.base_side^2 * P'.height) →
  ∃ (S : ℝ × ℝ × ℝ) (V : ℝ × ℝ × ℝ),
    S.2.2 = F.height / 2 + P'.height ∧
    V.2.2 = P.height ∧
    Real.sqrt ((S.1 - V.1)^2 + (S.2.1 - V.2.1)^2 + (S.2.2 - V.2.2)^2) = 2.5 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_frustum_theorem_l3911_391114


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3911_391125

theorem rectangle_area_change (L W : ℝ) (h : L * W = 540) :
  (0.8 * L) * (1.2 * W) = 518.4 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3911_391125


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_72_11_with_reversed_divisible_by_72_l3911_391186

/-- Reverses the digits of a given integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a five-digit integer -/
def isFiveDigit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

theorem smallest_five_digit_divisible_by_72_11_with_reversed_divisible_by_72 :
  ∀ p : ℕ,
    isFiveDigit p →
    p % 72 = 0 →
    (reverseDigits p) % 72 = 0 →
    p % 11 = 0 →
    p ≥ 80001 :=
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_72_11_with_reversed_divisible_by_72_l3911_391186


namespace NUMINAMATH_CALUDE_equation_solution_l3911_391151

def solution_set : Set (ℤ × ℤ) :=
  {(1, 12), (1, -12), (-9, 12), (-9, -12), (-4, 12), (-4, -12), (0, 0), (-8, 0), (-1, 0), (-7, 0)}

def satisfies_equation (p : ℤ × ℤ) : Prop :=
  let x := p.1
  let y := p.2
  x * (x + 1) * (x + 7) * (x + 8) = y^2

theorem equation_solution :
  ∀ p : ℤ × ℤ, satisfies_equation p ↔ p ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3911_391151


namespace NUMINAMATH_CALUDE_num_acceptance_configs_prove_num_acceptance_configs_l3911_391126

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the minimum number of companies -/
def min_companies : ℕ := 3

/-- Represents the acceptance configuration -/
structure AcceptanceConfig where
  student_acceptances : Fin num_students → ℕ
  company_acceptances : ℕ → Fin num_students → Bool
  each_student_diff : ∀ (i j : Fin num_students), i ≠ j → student_acceptances i ≠ student_acceptances j
  student_order : ∀ (i j : Fin num_students), i < j → student_acceptances i < student_acceptances j
  company_nonempty : ∀ (c : ℕ), c < min_companies → ∃ (s : Fin num_students), company_acceptances c s = true

/-- The main theorem stating the number of valid acceptance configurations -/
theorem num_acceptance_configs : (AcceptanceConfig → Prop) → ℕ := 60

/-- Proof of the theorem -/
theorem prove_num_acceptance_configs : num_acceptance_configs = 60 := by sorry

end NUMINAMATH_CALUDE_num_acceptance_configs_prove_num_acceptance_configs_l3911_391126
