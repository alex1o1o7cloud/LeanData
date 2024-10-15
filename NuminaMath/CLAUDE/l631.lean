import Mathlib

namespace NUMINAMATH_CALUDE_smallest_square_addition_l631_63116

theorem smallest_square_addition (n : ℕ) (h : n = 2020) : 
  ∃ k : ℕ, k = 1 ∧ 
  (∃ m : ℕ, (n - 1) * n * (n + 1) * (n + 2) + k = m^2) ∧
  (∀ j : ℕ, j < k → ¬∃ m : ℕ, (n - 1) * n * (n + 1) * (n + 2) + j = m^2) :=
by sorry

#check smallest_square_addition

end NUMINAMATH_CALUDE_smallest_square_addition_l631_63116


namespace NUMINAMATH_CALUDE_lowest_degree_is_four_l631_63197

/-- A polynomial with coefficients in ℤ -/
def IntPolynomial := Polynomial ℤ

/-- The set of coefficients of a polynomial -/
def coefficientSet (p : IntPolynomial) : Set ℤ :=
  {a : ℤ | ∃ (n : ℕ), p.coeff n = a}

/-- The property that a polynomial satisfies the given conditions -/
def satisfiesCondition (p : IntPolynomial) : Prop :=
  ∃ (b : ℤ),
    (∃ (a₁ : ℤ), a₁ ∈ coefficientSet p ∧ a₁ < b) ∧
    (∃ (a₂ : ℤ), a₂ ∈ coefficientSet p ∧ a₂ > b) ∧
    b ∉ coefficientSet p

/-- The theorem stating that the lowest degree of a polynomial satisfying the condition is 4 -/
theorem lowest_degree_is_four :
  ∃ (p : IntPolynomial),
    satisfiesCondition p ∧
    p.degree = 4 ∧
    ∀ (q : IntPolynomial), satisfiesCondition q → q.degree ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_lowest_degree_is_four_l631_63197


namespace NUMINAMATH_CALUDE_right_angle_in_triangle_l631_63108

theorem right_angle_in_triangle (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (A > 0 ∧ B > 0 ∧ C > 0) →
  (A + B + C = Real.pi) →
  -- Side lengths are positive
  (a > 0 ∧ b > 0 ∧ c > 0) →
  -- Given conditions
  (Real.sin B = Real.sin (2 * A)) →
  (c = 2 * a) →
  -- Conclusion
  C = Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_right_angle_in_triangle_l631_63108


namespace NUMINAMATH_CALUDE_box_volume_l631_63156

/-- Given a rectangular box with face areas 30, 18, and 45 square centimeters, 
    its volume is 90√3 cubic centimeters. -/
theorem box_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : b * c = 18) 
  (h3 : c * a = 45) : 
  a * b * c = 90 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l631_63156


namespace NUMINAMATH_CALUDE_tangent_and_intersection_l631_63134

-- Define the curve C
def C (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 - 9 * x^2 + 4

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := -12 * x + 8

-- Theorem statement
theorem tangent_and_intersection :
  -- The tangent line at x = 1 has the equation y = -12x + 8
  (∀ x, tangent_line x = -12 * x + 8) ∧
  -- The tangent line touches the curve at x = 1
  (C 1 = tangent_line 1) ∧
  -- The tangent line is indeed tangent to the curve at x = 1
  (deriv C 1 = -12) ∧
  -- The tangent line intersects the curve at two additional points
  (C (-2) = tangent_line (-2)) ∧
  (C (2/3) = tangent_line (2/3)) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_intersection_l631_63134


namespace NUMINAMATH_CALUDE_boat_journey_time_l631_63117

/-- Calculates the total journey time for a boat traveling upstream and downstream in a river -/
theorem boat_journey_time 
  (river_speed : ℝ) 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (h1 : river_speed = 2) 
  (h2 : boat_speed = 6) 
  (h3 : distance = 48) : 
  (distance / (boat_speed - river_speed)) + (distance / (boat_speed + river_speed)) = 18 :=
by sorry

end NUMINAMATH_CALUDE_boat_journey_time_l631_63117


namespace NUMINAMATH_CALUDE_empty_quadratic_set_l631_63145

theorem empty_quadratic_set (a : ℝ) :
  ({x : ℝ | a * x^2 - 2 * a * x + 1 < 0} = ∅) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_empty_quadratic_set_l631_63145


namespace NUMINAMATH_CALUDE_doughnut_machine_completion_time_l631_63120

-- Define the start time (6:00 AM) in minutes since midnight
def start_time : ℕ := 6 * 60

-- Define the time when one-fourth of the job is completed (9:00 AM) in minutes since midnight
def quarter_completion_time : ℕ := 9 * 60

-- Define the maintenance stop duration in minutes
def maintenance_duration : ℕ := 45

-- Define the completion time (6:45 PM) in minutes since midnight
def completion_time : ℕ := 18 * 60 + 45

-- Theorem statement
theorem doughnut_machine_completion_time :
  let working_duration := quarter_completion_time - start_time
  let total_duration := working_duration * 4 + maintenance_duration
  start_time + total_duration = completion_time :=
sorry

end NUMINAMATH_CALUDE_doughnut_machine_completion_time_l631_63120


namespace NUMINAMATH_CALUDE_black_haired_girls_count_l631_63165

theorem black_haired_girls_count (initial_total : ℕ) (added_blonde : ℕ) (initial_blonde : ℕ) : 
  initial_total = 80 → 
  added_blonde = 10 → 
  initial_blonde = 30 → 
  initial_total + added_blonde - (initial_blonde + added_blonde) = 50 := by
sorry

end NUMINAMATH_CALUDE_black_haired_girls_count_l631_63165


namespace NUMINAMATH_CALUDE_mountain_climbing_equivalence_l631_63178

/-- Given the elevations of two mountains and the number of times one is climbed,
    calculate how many times the other mountain needs to be climbed to cover the same distance. -/
theorem mountain_climbing_equivalence 
  (hugo_elevation : ℕ) 
  (elevation_difference : ℕ) 
  (hugo_climbs : ℕ) : 
  hugo_elevation = 10000 →
  elevation_difference = 2500 →
  hugo_climbs = 3 →
  (hugo_elevation * hugo_climbs) / (hugo_elevation - elevation_difference) = 4 := by
  sorry

end NUMINAMATH_CALUDE_mountain_climbing_equivalence_l631_63178


namespace NUMINAMATH_CALUDE_lines_parallel_perpendicular_l631_63119

/-- Two lines l₁ and l₂ in the plane --/
structure Lines (m : ℝ) where
  l₁ : ℝ → ℝ → ℝ := λ x y => 2*x + (m+1)*y + 4
  l₂ : ℝ → ℝ → ℝ := λ x y => m*x + 3*y - 6

/-- The lines are parallel --/
def parallel (m : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 2 = k * m ∧ m + 1 = k * 3 ∧ 4 ≠ k * (-6)

/-- The lines are perpendicular --/
def perpendicular (m : ℝ) : Prop :=
  2 * m + 3 * (m + 1) = 0

/-- Main theorem --/
theorem lines_parallel_perpendicular (m : ℝ) :
  (parallel m ↔ m = 2) ∧ (perpendicular m ↔ m = -3/5) := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_perpendicular_l631_63119


namespace NUMINAMATH_CALUDE_transistors_in_2010_l631_63110

/-- Moore's law tripling factor -/
def tripling_factor : ℕ := 3

/-- Years between tripling events -/
def years_per_tripling : ℕ := 3

/-- Initial number of transistors in 1995 -/
def initial_transistors : ℕ := 500000

/-- Years between 1995 and 2010 -/
def years_elapsed : ℕ := 15

/-- Number of tripling events in the given time period -/
def num_triplings : ℕ := years_elapsed / years_per_tripling

/-- Calculates the number of transistors after a given number of tripling events -/
def transistors_after_triplings (initial : ℕ) (triplings : ℕ) : ℕ :=
  initial * tripling_factor ^ triplings

/-- Theorem: The number of transistors in 2010 is 121,500,000 -/
theorem transistors_in_2010 :
  transistors_after_triplings initial_transistors num_triplings = 121500000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_in_2010_l631_63110


namespace NUMINAMATH_CALUDE_linda_travel_distance_l631_63130

/-- Represents the travel data for one day --/
structure DayTravel where
  totalTime : ℕ
  timePerMile : ℕ

/-- Calculates the distance traveled in a day --/
def distanceTraveled (day : DayTravel) : ℚ :=
  day.totalTime / day.timePerMile

/-- Represents Linda's travel data over three days --/
structure ThreeDayTravel where
  day1 : DayTravel
  day2 : DayTravel
  day3 : DayTravel

/-- The main theorem to prove --/
theorem linda_travel_distance 
  (travel : ThreeDayTravel)
  (time_condition : travel.day1.totalTime = 60 ∧ 
                    travel.day2.totalTime = 75 ∧ 
                    travel.day3.totalTime = 90)
  (time_increase : travel.day2.timePerMile = travel.day1.timePerMile + 3 ∧
                   travel.day3.timePerMile = travel.day2.timePerMile + 3)
  (integer_distance : ∀ d : DayTravel, d ∈ [travel.day1, travel.day2, travel.day3] → 
                      (distanceTraveled d).den = 1)
  (integer_time : ∀ d : DayTravel, d ∈ [travel.day1, travel.day2, travel.day3] → 
                  d.timePerMile > 0) :
  (distanceTraveled travel.day1 + distanceTraveled travel.day2 + distanceTraveled travel.day3 : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_linda_travel_distance_l631_63130


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_power_l631_63173

theorem sqrt_seven_to_sixth_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_power_l631_63173


namespace NUMINAMATH_CALUDE_batsman_average_increase_l631_63112

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the new average after an additional inning -/
def newAverage (stats : BatsmanStats) (newScore : ℕ) : ℚ :=
  (stats.totalScore + newScore) / (stats.innings + 1)

/-- Theorem: If a batsman's average increases by 1 after scoring 69 in the 11th inning,
    then the new average is 59 -/
theorem batsman_average_increase (stats : BatsmanStats) :
  stats.innings = 10 →
  newAverage stats 69 = stats.average + 1 →
  newAverage stats 69 = 59 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l631_63112


namespace NUMINAMATH_CALUDE_expression_factorization_l631_63179

theorem expression_factorization (x : ℝ) :
  (20 * x^3 + 100 * x - 10) - (-3 * x^3 + 5 * x - 15) = 5 * (23 * x^3 + 19 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l631_63179


namespace NUMINAMATH_CALUDE_tangent_length_fq_l631_63199

-- Define the triangle
structure RightTriangle where
  de : ℝ
  df : ℝ
  ef : ℝ
  right_angle_at_e : de^2 + ef^2 = df^2

-- Define the circle
structure TangentCircle where
  center_on_de : Bool
  tangent_to_df : Bool
  tangent_to_ef : Bool

-- Theorem statement
theorem tangent_length_fq 
  (t : RightTriangle) 
  (c : TangentCircle) 
  (h1 : t.de = 7) 
  (h2 : t.df = Real.sqrt 85) 
  (h3 : c.center_on_de = true) 
  (h4 : c.tangent_to_df = true) 
  (h5 : c.tangent_to_ef = true) : 
  ∃ q : ℝ, q = 6 ∧ q = t.ef := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_fq_l631_63199


namespace NUMINAMATH_CALUDE_arrangement_count_l631_63151

def number_of_arrangements (total_people : ℕ) (selected_people : ℕ) 
  (meeting_a_participants : ℕ) (meeting_b_participants : ℕ) (meeting_c_participants : ℕ) : ℕ :=
  Nat.choose total_people selected_people * 
  Nat.choose selected_people meeting_a_participants * 
  Nat.choose (selected_people - meeting_a_participants) meeting_b_participants

theorem arrangement_count : 
  number_of_arrangements 10 4 2 1 1 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l631_63151


namespace NUMINAMATH_CALUDE_units_digit_of_G_1000_l631_63121

-- Define G_n
def G (n : ℕ) : ℕ := 3^(3^n) + 1

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_G_1000 : unitsDigit (G 1000) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_G_1000_l631_63121


namespace NUMINAMATH_CALUDE_geometric_sum_n_eq_1_l631_63138

theorem geometric_sum_n_eq_1 (a : ℝ) (h : a ≠ 1) :
  1 + a = (1 - a^3) / (1 - a) := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_n_eq_1_l631_63138


namespace NUMINAMATH_CALUDE_area_at_stage_8_l631_63129

/-- The side length of each square in inches -/
def square_side : ℝ := 4

/-- The number of squares at a given stage -/
def num_squares (stage : ℕ) : ℕ := stage

/-- The area of the rectangle at a given stage in square inches -/
def rectangle_area (stage : ℕ) : ℝ :=
  (num_squares stage) * (square_side ^ 2)

/-- Theorem: The area of the rectangle at Stage 8 is 128 square inches -/
theorem area_at_stage_8 : rectangle_area 8 = 128 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_8_l631_63129


namespace NUMINAMATH_CALUDE_q_share_is_7200_l631_63161

/-- Calculates the share of profit for a partner in a business partnership. -/
def calculateShareOfProfit (investment1 : ℕ) (investment2 : ℕ) (totalProfit : ℕ) : ℕ :=
  let totalInvestment := investment1 + investment2
  (investment2 * totalProfit) / totalInvestment

/-- Theorem stating that Q's share of the profit is 7200 given the specified investments and total profit. -/
theorem q_share_is_7200 :
  calculateShareOfProfit 54000 36000 18000 = 7200 := by
  sorry

#eval calculateShareOfProfit 54000 36000 18000

end NUMINAMATH_CALUDE_q_share_is_7200_l631_63161


namespace NUMINAMATH_CALUDE_train_passing_platform_l631_63180

/-- Given a train of length 250 meters passing a pole in 10 seconds,
    prove that it takes 60 seconds to pass a platform of length 1250 meters. -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (pole_passing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 250)
  (h2 : pole_passing_time = 10)
  (h3 : platform_length = 1250) :
  (train_length + platform_length) / (train_length / pole_passing_time) = 60 := by
  sorry

#check train_passing_platform

end NUMINAMATH_CALUDE_train_passing_platform_l631_63180


namespace NUMINAMATH_CALUDE_ice_cream_arrangement_count_l631_63160

theorem ice_cream_arrangement_count : Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_arrangement_count_l631_63160


namespace NUMINAMATH_CALUDE_problem_solution_l631_63144

def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1) 1, x^2 - x - m < 0}

def A (a : ℝ) : Set ℝ := {x | (x - 3*a) * (x - a - 2) < 0}

theorem problem_solution :
  (B = Set.Ioi 2) ∧
  ({a : ℝ | A a ⊆ B ∧ A a ≠ B} = Set.Ici (2/3)) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l631_63144


namespace NUMINAMATH_CALUDE_sally_pens_taken_home_l631_63148

def total_pens : ℕ := 5230
def num_students : ℕ := 89
def pens_per_student : ℕ := 58

def pens_distributed : ℕ := num_students * pens_per_student
def pens_remaining : ℕ := total_pens - pens_distributed
def pens_in_locker : ℕ := pens_remaining / 2
def pens_taken_home : ℕ := pens_remaining - pens_in_locker

theorem sally_pens_taken_home : pens_taken_home = 34 := by
  sorry

end NUMINAMATH_CALUDE_sally_pens_taken_home_l631_63148


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_l631_63154

theorem largest_power_of_two_dividing_difference : ∃ (k : ℕ), k = 13 ∧ 
  (∀ (n : ℕ), 2^n ∣ (10^10 - 2^10) → n ≤ k) ∧ 
  (2^k ∣ (10^10 - 2^10)) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_l631_63154


namespace NUMINAMATH_CALUDE_horner_v2_at_2_l631_63167

def horner_polynomial (x : ℝ) : ℝ := 2*x^7 + x^6 + x^4 + x^2 + 1

def horner_v2 (x : ℝ) : ℝ := 
  let v0 := 2
  let v1 := 2*x + 1
  v1 * x

theorem horner_v2_at_2 : horner_v2 2 = 10 := by
  sorry

#eval horner_v2 2

end NUMINAMATH_CALUDE_horner_v2_at_2_l631_63167


namespace NUMINAMATH_CALUDE_midpoint_triangle_is_equilateral_l631_63139

-- Define the points in the plane
variable (A B C D E F G M N P : ℝ × ℝ)

-- Define the conditions
def is_midpoint (M A B : ℝ × ℝ) : Prop := M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def is_equilateral_triangle (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 ∧
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = (Z.1 - X.1)^2 + (Z.2 - X.2)^2

-- State the theorem
theorem midpoint_triangle_is_equilateral
  (h1 : is_midpoint M A B)
  (h2 : is_midpoint P G F)
  (h3 : is_midpoint N E F)
  (h4 : is_equilateral_triangle B C E)
  (h5 : is_equilateral_triangle C D F)
  (h6 : is_equilateral_triangle D A G) :
  is_equilateral_triangle M N P :=
sorry

end NUMINAMATH_CALUDE_midpoint_triangle_is_equilateral_l631_63139


namespace NUMINAMATH_CALUDE_compound_interest_rate_l631_63177

/-- Given an initial amount P at compound interest that sums to 17640 after 2 years
    and 22050 after 3 years, the annual interest rate is 25%. -/
theorem compound_interest_rate (P : ℝ) : 
  P * (1 + 0.25)^2 = 17640 ∧ P * (1 + 0.25)^3 = 22050 → 0.25 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l631_63177


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l631_63168

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - 5*x + 6 ≤ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l631_63168


namespace NUMINAMATH_CALUDE_seventh_term_is_four_l631_63183

/-- A geometric sequence with first term 1 and a specific condition on terms 3, 4, and 5 -/
def special_geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, ∃ r : ℝ, ∀ k : ℕ, a (k + 1) = a k * r) ∧  -- geometric sequence condition
  a 1 = 1 ∧                                           -- first term is 1
  a 3 * a 5 = 4 * (a 4 - 1)                           -- given condition

/-- The 7th term of the special geometric sequence is 4 -/
theorem seventh_term_is_four (a : ℕ → ℝ) (h : special_geometric_sequence a) : a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_four_l631_63183


namespace NUMINAMATH_CALUDE_problem_statement_l631_63159

theorem problem_statement (a b : ℤ) (h1 : 6 * a + 3 * b = 0) (h2 : a = b - 3) : 5 * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l631_63159


namespace NUMINAMATH_CALUDE_sum_A_B_equals_24_l631_63146

theorem sum_A_B_equals_24 (A B : ℚ) (h1 : (1 : ℚ) / 6 * (1 : ℚ) / 3 = 1 / (A * 3))
  (h2 : (1 : ℚ) / 6 * (1 : ℚ) / 3 = 1 / B) : A + B = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_A_B_equals_24_l631_63146


namespace NUMINAMATH_CALUDE_total_chocolate_bars_l631_63123

/-- Represents the number of chocolate bars in a large box -/
def chocolateBarsInLargeBox (smallBoxes : ℕ) (barsPerSmallBox : ℕ) : ℕ :=
  smallBoxes * barsPerSmallBox

/-- Proves that the total number of chocolate bars in the large box is 500 -/
theorem total_chocolate_bars :
  chocolateBarsInLargeBox 20 25 = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_chocolate_bars_l631_63123


namespace NUMINAMATH_CALUDE_three_digit_sum_property_l631_63172

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_sum_property (n : ℕ) :
  100 ≤ n ∧ n < 1000 ∧ 
  digit_sum n = 3 * digit_sum (n - 75) →
  n = 189 ∨ n = 675 := by
sorry

end NUMINAMATH_CALUDE_three_digit_sum_property_l631_63172


namespace NUMINAMATH_CALUDE_table_length_is_77_l631_63140

/-- Represents the dimensions of a rectangular table. -/
structure TableDimensions where
  length : ℕ
  width : ℕ

/-- Represents the dimensions of a paper sheet. -/
structure SheetDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the length of a table given its width and the dimensions of the paper sheets used to cover it. -/
def calculateTableLength (tableWidth : ℕ) (sheet : SheetDimensions) : ℕ :=
  sheet.length + (tableWidth - sheet.width)

/-- Theorem stating that for a table of width 80 cm covered with 5x8 cm sheets,
    where each sheet is placed 1 cm higher and 1 cm to the right of the previous one,
    the length of the table is 77 cm. -/
theorem table_length_is_77 :
  let tableWidth : ℕ := 80
  let sheet : SheetDimensions := ⟨5, 8⟩
  calculateTableLength tableWidth sheet = 77 := by
  sorry

#check table_length_is_77

end NUMINAMATH_CALUDE_table_length_is_77_l631_63140


namespace NUMINAMATH_CALUDE_range_of_a_for_local_max_l631_63113

noncomputable def f (a b x : ℝ) : ℝ := Real.log x + a * x^2 + b * x

theorem range_of_a_for_local_max (a b : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f a b x ≤ f a b 1) →
  a < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_local_max_l631_63113


namespace NUMINAMATH_CALUDE_expand_expression_l631_63155

theorem expand_expression (x y z : ℝ) : 
  (x + 5) * (3 * y + 2 * z + 15) = 3 * x * y + 2 * x * z + 15 * x + 15 * y + 10 * z + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l631_63155


namespace NUMINAMATH_CALUDE_defective_pens_l631_63190

theorem defective_pens (total_pens : ℕ) (prob_non_defective : ℚ) : 
  total_pens = 12 →
  prob_non_defective = 7/33 →
  (∃ (defective : ℕ), 
    defective ≤ total_pens ∧ 
    (total_pens - defective : ℚ) / total_pens * ((total_pens - defective - 1) : ℚ) / (total_pens - 1) = prob_non_defective ∧
    defective = 4) := by
  sorry

end NUMINAMATH_CALUDE_defective_pens_l631_63190


namespace NUMINAMATH_CALUDE_students_present_l631_63126

theorem students_present (total : ℕ) (absent_percent : ℚ) (present : ℕ) : 
  total = 50 → 
  absent_percent = 1/10 → 
  present = total - (total * (absent_percent : ℚ)).num / (absent_percent : ℚ).den → 
  present = 45 := by sorry

end NUMINAMATH_CALUDE_students_present_l631_63126


namespace NUMINAMATH_CALUDE_equation_solution_l631_63147

theorem equation_solution : 
  ∃ x : ℚ, (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3 ∧ x = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l631_63147


namespace NUMINAMATH_CALUDE_expression_simplification_l631_63104

theorem expression_simplification (m : ℝ) (h1 : m ≠ 2) (h2 : m ≠ -3) :
  (m - (4*m - 9) / (m - 2)) / ((m^2 - 9) / (m - 2)) = (m - 3) / (m + 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l631_63104


namespace NUMINAMATH_CALUDE_meat_voters_count_l631_63192

/-- The number of students who voted for veggies -/
def veggies_votes : ℕ := 337

/-- The total number of students who voted -/
def total_votes : ℕ := 672

/-- The number of students who voted for meat -/
def meat_votes : ℕ := total_votes - veggies_votes

theorem meat_voters_count : meat_votes = 335 := by
  sorry

end NUMINAMATH_CALUDE_meat_voters_count_l631_63192


namespace NUMINAMATH_CALUDE_mrs_petersons_change_l631_63188

theorem mrs_petersons_change (number_of_tumblers : ℕ) (price_per_tumbler : ℕ) (number_of_bills : ℕ) (bill_value : ℕ) : 
  number_of_tumblers = 10 →
  price_per_tumbler = 45 →
  number_of_bills = 5 →
  bill_value = 100 →
  (number_of_bills * bill_value) - (number_of_tumblers * price_per_tumbler) = 50 :=
by
  sorry

#check mrs_petersons_change

end NUMINAMATH_CALUDE_mrs_petersons_change_l631_63188


namespace NUMINAMATH_CALUDE_julia_normal_mile_time_l631_63135

/-- Represents Julia's running times -/
structure JuliaRunningTimes where
  normalMileTime : ℝ
  newShoesMileTime : ℝ

/-- The conditions of the problem -/
def problemConditions (j : JuliaRunningTimes) : Prop :=
  j.newShoesMileTime = 13 ∧
  5 * j.newShoesMileTime = 5 * j.normalMileTime + 15

/-- The theorem stating Julia's normal mile time -/
theorem julia_normal_mile_time (j : JuliaRunningTimes) 
  (h : problemConditions j) : j.normalMileTime = 10 := by
  sorry


end NUMINAMATH_CALUDE_julia_normal_mile_time_l631_63135


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_2022_l631_63100

theorem smallest_n_divisible_by_2022 :
  ∃ (n : ℕ), n > 1 ∧ n^7 - 1 % 2022 = 0 ∧
  ∀ (m : ℕ), m > 1 ∧ m < n → m^7 - 1 % 2022 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_2022_l631_63100


namespace NUMINAMATH_CALUDE_natasha_quarters_l631_63157

theorem natasha_quarters (q : ℕ) : 
  (10 < (q : ℚ) * (1/4) ∧ (q : ℚ) * (1/4) < 200) ∧ 
  q % 4 = 2 ∧ q % 5 = 2 ∧ q % 6 = 2 ↔ 
  ∃ k : ℕ, k ≥ 1 ∧ k ≤ 13 ∧ q = 60 * k + 2 :=
by sorry

end NUMINAMATH_CALUDE_natasha_quarters_l631_63157


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l631_63186

/-- Represents the number of accent options for each letter in "cesontoiseaux" --/
def accentOptions : List Nat := [2, 5, 5, 1, 1, 3, 3, 1, 1, 2, 3, 1, 4]

/-- The number of ways to split 12 letters into 3 words --/
def wordSplitOptions : Nat := 66

/-- Calculates the total number of possible phrases --/
def totalPhrases : Nat :=
  wordSplitOptions * (accentOptions.foldl (·*·) 1)

/-- Theorem stating that the number of distinct prime factors of totalPhrases is 4 --/
theorem distinct_prime_factors_count :
  (Nat.factors totalPhrases).toFinset.card = 4 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l631_63186


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_sum_l631_63150

/-- The sum of binary digits of a positive integer -/
def s (n : ℕ+) : ℕ := sorry

/-- The sum of s(n) for n from 1 to 2^k -/
def S (k : ℕ+) : ℕ :=
  Finset.sum (Finset.range (2^k.val)) (fun i => s ⟨i + 1, Nat.succ_pos i⟩)

/-- The main theorem: S(k) = 2^(k-1) * k + 1 for all positive integers k -/
theorem sum_of_binary_digits_sum (k : ℕ+) : S k = 2^(k.val - 1) * k.val + 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_sum_l631_63150


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l631_63103

theorem division_multiplication_problem : 377 / 13 / 29 * (1 / 4) / 2 = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l631_63103


namespace NUMINAMATH_CALUDE_robie_cards_count_l631_63195

theorem robie_cards_count (cards_per_box : ℕ) (unboxed_cards : ℕ) (boxes_given_away : ℕ) (boxes_remaining : ℕ) : 
  cards_per_box = 25 →
  unboxed_cards = 11 →
  boxes_given_away = 6 →
  boxes_remaining = 12 →
  cards_per_box * (boxes_given_away + boxes_remaining) + unboxed_cards = 461 := by
sorry

end NUMINAMATH_CALUDE_robie_cards_count_l631_63195


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l631_63174

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + k * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + k * y + 12 = 0 → y = x) → 
  k = 12 ∨ k = -12 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l631_63174


namespace NUMINAMATH_CALUDE_triangle_side_length_l631_63109

/-- An equilateral triangle with a point inside and perpendiculars to its sides. -/
structure TriangleWithPoint where
  /-- Side length of the equilateral triangle -/
  side_length : ℝ
  /-- Distance from the point to side AB -/
  dist_to_AB : ℝ
  /-- Distance from the point to side BC -/
  dist_to_BC : ℝ
  /-- Distance from the point to side CA -/
  dist_to_CA : ℝ
  /-- The triangle is equilateral -/
  equilateral : side_length > 0
  /-- The point is inside the triangle -/
  point_inside : dist_to_AB > 0 ∧ dist_to_BC > 0 ∧ dist_to_CA > 0

/-- Theorem: If the perpendicular distances are 2, 2√2, and 4, then the side length is 4√3 + (4√6)/3 -/
theorem triangle_side_length (t : TriangleWithPoint) 
  (h1 : t.dist_to_AB = 2) 
  (h2 : t.dist_to_BC = 2 * Real.sqrt 2) 
  (h3 : t.dist_to_CA = 4) : 
  t.side_length = 4 * Real.sqrt 3 + (4 * Real.sqrt 6) / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l631_63109


namespace NUMINAMATH_CALUDE_composition_difference_constant_l631_63164

/-- Given two functions f and g, prove that their composition difference is constant. -/
theorem composition_difference_constant (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 4 * x - 5) 
  (hg : ∀ x, g x = x / 4 + 1) : 
  ∀ x, f (g x) - g (f x) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_composition_difference_constant_l631_63164


namespace NUMINAMATH_CALUDE_lawrence_county_kids_count_l631_63185

theorem lawrence_county_kids_count :
  let kids_stayed_home : ℕ := 644997
  let kids_went_to_camp : ℕ := 893835
  let outside_kids_at_camp : ℕ := 78
  kids_stayed_home + kids_went_to_camp = 1538832 :=
by sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_count_l631_63185


namespace NUMINAMATH_CALUDE_clothing_colors_l631_63175

-- Define the colors
inductive Color
| Red
| Blue

-- Define a structure for a child's clothing
structure Clothing where
  tshirt : Color
  shorts : Color

-- Define the four children
def Alyna : Clothing := sorry
def Bohdan : Clothing := sorry
def Vika : Clothing := sorry
def Grysha : Clothing := sorry

-- Define the theorem
theorem clothing_colors :
  -- Conditions
  (Alyna.tshirt = Color.Red) →
  (Bohdan.tshirt = Color.Red) →
  (Alyna.shorts ≠ Bohdan.shorts) →
  (Vika.tshirt ≠ Grysha.tshirt) →
  (Vika.shorts = Color.Blue) →
  (Grysha.shorts = Color.Blue) →
  (Alyna.tshirt ≠ Vika.tshirt) →
  (Alyna.shorts ≠ Vika.shorts) →
  -- Conclusion
  (Alyna = ⟨Color.Red, Color.Red⟩ ∧
   Bohdan = ⟨Color.Red, Color.Blue⟩ ∧
   Vika = ⟨Color.Blue, Color.Blue⟩ ∧
   Grysha = ⟨Color.Red, Color.Blue⟩) :=
by
  sorry


end NUMINAMATH_CALUDE_clothing_colors_l631_63175


namespace NUMINAMATH_CALUDE_solve_for_x_l631_63181

theorem solve_for_x (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : y = 1) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l631_63181


namespace NUMINAMATH_CALUDE_second_to_first_l631_63122

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate for a point being in the second quadrant -/
def inSecondQuadrant (p : Point) : Prop := p.x < 0 ∧ p.y > 0

/-- Predicate for a point being in the first quadrant -/
def inFirstQuadrant (p : Point) : Prop := p.x > 0 ∧ p.y > 0

/-- Theorem: If A(m,n) is in the second quadrant, then B(-m,|n|) is in the first quadrant -/
theorem second_to_first (m n : ℝ) :
  inSecondQuadrant ⟨m, n⟩ → inFirstQuadrant ⟨-m, |n|⟩ := by
  sorry

end NUMINAMATH_CALUDE_second_to_first_l631_63122


namespace NUMINAMATH_CALUDE_adult_tickets_count_l631_63196

theorem adult_tickets_count
  (adult_price : ℝ)
  (child_price : ℝ)
  (total_tickets : ℕ)
  (total_cost : ℝ)
  (h1 : adult_price = 5.5)
  (h2 : child_price = 3.5)
  (h3 : total_tickets = 21)
  (h4 : total_cost = 83.5) :
  ∃ (adult_count : ℕ) (child_count : ℕ),
    adult_count + child_count = total_tickets ∧
    adult_count * adult_price + child_count * child_price = total_cost ∧
    adult_count = 5 :=
by sorry

end NUMINAMATH_CALUDE_adult_tickets_count_l631_63196


namespace NUMINAMATH_CALUDE_probability_between_R_and_S_l631_63106

/-- Given a line segment PQ with points R and S, where PQ = 4PR and PQ = 8QR,
    the probability that a randomly selected point on PQ lies between R and S is 5/8. -/
theorem probability_between_R_and_S (P Q R S : Real) (h1 : Q - P = 4 * (R - P)) (h2 : Q - P = 8 * (Q - R)) :
  (S - R) / (Q - P) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_between_R_and_S_l631_63106


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l631_63115

/-- Compound interest calculation --/
theorem compound_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℕ) 
  (h1 : principal = 5000)
  (h2 : rate = 0.1)
  (h3 : time = 2) :
  principal * (1 + rate) ^ time = 6050 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l631_63115


namespace NUMINAMATH_CALUDE_a_upper_bound_l631_63182

theorem a_upper_bound (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2) 3, 2*x > x^2 + a) → a < -8 := by
  sorry

end NUMINAMATH_CALUDE_a_upper_bound_l631_63182


namespace NUMINAMATH_CALUDE_garden_planting_area_l631_63125

def garden_length : ℝ := 18
def garden_width : ℝ := 14
def pond_length : ℝ := 4
def pond_width : ℝ := 2
def flower_bed_base : ℝ := 3
def flower_bed_height : ℝ := 2

theorem garden_planting_area :
  garden_length * garden_width - (pond_length * pond_width + 1/2 * flower_bed_base * flower_bed_height) = 241 := by
  sorry

end NUMINAMATH_CALUDE_garden_planting_area_l631_63125


namespace NUMINAMATH_CALUDE_initial_balance_was_800_liza_initial_balance_l631_63198

/-- Represents the transactions in Liza's checking account --/
structure AccountTransactions where
  initial_balance : ℕ
  rent_payment : ℕ
  paycheck_deposit : ℕ
  electricity_bill : ℕ
  internet_bill : ℕ
  phone_bill : ℕ
  final_balance : ℕ

/-- Theorem stating that given the transactions and final balance, the initial balance was 800 --/
theorem initial_balance_was_800 (t : AccountTransactions) 
  (h1 : t.rent_payment = 450)
  (h2 : t.paycheck_deposit = 1500)
  (h3 : t.electricity_bill = 117)
  (h4 : t.internet_bill = 100)
  (h5 : t.phone_bill = 70)
  (h6 : t.final_balance = 1563)
  (h7 : t.initial_balance - t.rent_payment + t.paycheck_deposit - t.electricity_bill - t.internet_bill - t.phone_bill = t.final_balance) :
  t.initial_balance = 800 := by
  sorry

/-- Main theorem that proves Liza had $800 in her checking account on Tuesday --/
theorem liza_initial_balance : ∃ (t : AccountTransactions), t.initial_balance = 800 ∧ 
  t.rent_payment = 450 ∧
  t.paycheck_deposit = 1500 ∧
  t.electricity_bill = 117 ∧
  t.internet_bill = 100 ∧
  t.phone_bill = 70 ∧
  t.final_balance = 1563 ∧
  t.initial_balance - t.rent_payment + t.paycheck_deposit - t.electricity_bill - t.internet_bill - t.phone_bill = t.final_balance := by
  sorry

end NUMINAMATH_CALUDE_initial_balance_was_800_liza_initial_balance_l631_63198


namespace NUMINAMATH_CALUDE_rectangle_longest_side_l631_63101

/-- Given a rectangle with perimeter 240 feet and area equal to eight times its perimeter,
    the length of its longest side is 101 feet. -/
theorem rectangle_longest_side : ∀ l w : ℝ,
  (2 * l + 2 * w = 240) →  -- perimeter is 240 feet
  (l * w = 8 * 240) →      -- area is 8 times the perimeter
  (l ≥ 0 ∧ w ≥ 0) →        -- length and width are non-negative
  (max l w = 101) :=       -- the longest side is 101 feet
by sorry

end NUMINAMATH_CALUDE_rectangle_longest_side_l631_63101


namespace NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_l631_63153

/-- Represents a line in a plane -/
structure Line where
  -- Add necessary fields here
  
/-- Represents an angle in a plane -/
structure Angle where
  -- Add necessary fields here

/-- Represents a plane -/
structure Plane where
  -- Add necessary fields here

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  sorry

/-- A line intersects two other lines -/
def intersects (l : Line) (l1 l2 : Line) : Prop :=
  sorry

/-- Two angles are corresponding angles -/
def corresponding_angles (a1 a2 : Angle) (l1 l2 l : Line) : Prop :=
  sorry

/-- Two angles are equal -/
def angles_equal (a1 a2 : Angle) : Prop :=
  sorry

/-- Two angles are supplementary -/
def angles_supplementary (a1 a2 : Angle) : Prop :=
  sorry

/-- Main theorem: If two lines are parallel and intersected by a transversal,
    then the corresponding angles are either equal or supplementary -/
theorem parallel_lines_corresponding_angles 
  (p : Plane) (l1 l2 l : Line) (a1 a2 : Angle) :
  parallel l1 l2 → intersects l l1 l2 → corresponding_angles a1 a2 l1 l2 l →
  angles_equal a1 a2 ∨ angles_supplementary a1 a2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_l631_63153


namespace NUMINAMATH_CALUDE_equal_angles_in_special_quadrilateral_l631_63194

/-- A point on the Cartesian plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- A quadrilateral on the Cartesian plane -/
structure Quadrilateral := (A B C D : Point)

/-- Checks if a point is on the hyperbola y = 1/x -/
def on_hyperbola (p : Point) : Prop := p.y = 1 / p.x

/-- Checks if a point is on the negative branch of the hyperbola -/
def on_negative_branch (p : Point) : Prop := on_hyperbola p ∧ p.x < 0

/-- Checks if a point is on the positive branch of the hyperbola -/
def on_positive_branch (p : Point) : Prop := on_hyperbola p ∧ p.x > 0

/-- Checks if a point is to the left of another point -/
def left_of (p1 p2 : Point) : Prop := p1.x < p2.x

/-- Checks if a line segment passes through the origin -/
def passes_through_origin (p1 p2 : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ t * p1.x + (1 - t) * p2.x = 0 ∧ t * p1.y + (1 - t) * p2.y = 0

/-- Calculates the angle between two lines given three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Main theorem -/
theorem equal_angles_in_special_quadrilateral (ABCD : Quadrilateral) :
  on_negative_branch ABCD.A →
  on_negative_branch ABCD.D →
  on_positive_branch ABCD.B →
  on_positive_branch ABCD.C →
  left_of ABCD.B ABCD.C →
  passes_through_origin ABCD.A ABCD.C →
  angle ABCD.B ABCD.A ABCD.D = angle ABCD.B ABCD.C ABCD.D :=
by sorry

end NUMINAMATH_CALUDE_equal_angles_in_special_quadrilateral_l631_63194


namespace NUMINAMATH_CALUDE_some_number_value_l631_63171

theorem some_number_value : 
  ∀ some_number : ℝ, 
  (some_number * 3.6) / (0.04 * 0.1 * 0.007) = 990.0000000000001 → 
  some_number = 7.7 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l631_63171


namespace NUMINAMATH_CALUDE_prob_at_least_one_white_l631_63118

/-- The number of red balls in the bag -/
def num_red : ℕ := 3

/-- The number of white balls in the bag -/
def num_white : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red + num_white

/-- The number of balls drawn -/
def num_drawn : ℕ := 2

/-- The probability of drawing at least one white ball when selecting two balls -/
theorem prob_at_least_one_white :
  (1 : ℚ) - (num_red.choose num_drawn : ℚ) / (total_balls.choose num_drawn : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_white_l631_63118


namespace NUMINAMATH_CALUDE_permutation_count_mod_500_l631_63162

/-- Represents the number of ways to arrange letters in specific positions -/
def arrange (n m : ℕ) : ℕ := Nat.choose n m

/-- Calculates the sum of products of arrangements for different k values -/
def sum_arrangements : ℕ :=
  (arrange 5 1 * arrange 6 0 * arrange 7 2) +
  (arrange 5 2 * arrange 6 1 * arrange 7 3) +
  (arrange 5 3 * arrange 6 2 * arrange 7 4) +
  (arrange 5 4 * arrange 6 3 * arrange 7 5) +
  (arrange 5 5 * arrange 6 4 * arrange 7 6)

/-- The main theorem stating the result of the permutation count modulo 500 -/
theorem permutation_count_mod_500 :
  sum_arrangements % 500 = 160 := by sorry

end NUMINAMATH_CALUDE_permutation_count_mod_500_l631_63162


namespace NUMINAMATH_CALUDE_complex_sum_powers_of_i_l631_63187

theorem complex_sum_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ i + i^2 + i^3 + i^4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_powers_of_i_l631_63187


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_l631_63137

-- Define the function
def f (x : ℝ) : ℝ := x^2 + x - 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 2*x + 1

-- Theorem statement
theorem tangent_line_at_point_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let k : ℝ := f' x₀
  ∀ x y : ℝ, (k * (x - x₀) = y - y₀) ↔ (3*x - y - 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_l631_63137


namespace NUMINAMATH_CALUDE_negative_324_same_terminal_side_as_36_l631_63193

/-- Two angles have the same terminal side if their difference is a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

/-- The angle -324° has the same terminal side as 36° -/
theorem negative_324_same_terminal_side_as_36 :
  same_terminal_side 36 (-324) := by
  sorry

end NUMINAMATH_CALUDE_negative_324_same_terminal_side_as_36_l631_63193


namespace NUMINAMATH_CALUDE_cos_240_degrees_l631_63191

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l631_63191


namespace NUMINAMATH_CALUDE_joe_fish_compared_to_sam_l631_63141

theorem joe_fish_compared_to_sam (harry_fish joe_fish sam_fish : ℕ) 
  (harry_joe_ratio : harry_fish = 4 * joe_fish)
  (joe_sam_ratio : ∃ x : ℕ, joe_fish = x * sam_fish)
  (sam_fish_count : sam_fish = 7)
  (harry_fish_count : harry_fish = 224) :
  ∃ x : ℕ, joe_fish = 8 * sam_fish := by
  sorry

end NUMINAMATH_CALUDE_joe_fish_compared_to_sam_l631_63141


namespace NUMINAMATH_CALUDE_sally_pokemon_cards_sally_pokemon_cards_proof_l631_63132

theorem sally_pokemon_cards : ℕ → Prop :=
  fun x =>
    let sally_initial : ℕ := 27
    let dan_cards : ℕ := 41
    let difference : ℕ := 6
    sally_initial + x = dan_cards + difference →
    x = 20

-- The proof is omitted
theorem sally_pokemon_cards_proof : ∃ x, sally_pokemon_cards x :=
  sorry

end NUMINAMATH_CALUDE_sally_pokemon_cards_sally_pokemon_cards_proof_l631_63132


namespace NUMINAMATH_CALUDE_angle_B_when_A_is_pi_sixth_sin_A_plus_sin_C_range_l631_63131

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a = t.b * Real.tan t.A ∧
  t.B > Real.pi / 2

-- Theorem 1
theorem angle_B_when_A_is_pi_sixth (t : Triangle) 
  (h : is_valid_triangle t) (h_A : t.A = Real.pi / 6) : 
  t.B = 2 * Real.pi / 3 := 
sorry

-- Theorem 2
theorem sin_A_plus_sin_C_range (t : Triangle) 
  (h : is_valid_triangle t) : 
  Real.sqrt 2 / 2 < Real.sin t.A + Real.sin t.C ∧ 
  Real.sin t.A + Real.sin t.C ≤ 9 / 8 := 
sorry

end NUMINAMATH_CALUDE_angle_B_when_A_is_pi_sixth_sin_A_plus_sin_C_range_l631_63131


namespace NUMINAMATH_CALUDE_canvas_bag_lower_carbon_l631_63143

/-- The number of shopping trips required for a canvas bag to be the lower-carbon solution -/
def shopping_trips_for_lower_carbon (canvas_co2_pounds : ℕ) (plastic_co2_ounces : ℕ) (bags_per_trip : ℕ) : ℕ :=
  let canvas_co2_ounces : ℕ := canvas_co2_pounds * 16
  let plastic_co2_per_trip : ℕ := plastic_co2_ounces * bags_per_trip
  canvas_co2_ounces / plastic_co2_per_trip

/-- Theorem stating the number of shopping trips required for the canvas bag to be lower-carbon -/
theorem canvas_bag_lower_carbon :
  shopping_trips_for_lower_carbon 600 4 8 = 300 := by
  sorry

end NUMINAMATH_CALUDE_canvas_bag_lower_carbon_l631_63143


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l631_63107

theorem count_integers_satisfying_inequality : 
  ∃ (S : Finset Int), (∀ n : Int, n ∈ S ↔ (n - 3) * (n + 5) < 0) ∧ Finset.card S = 7 :=
sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l631_63107


namespace NUMINAMATH_CALUDE_sixteen_integer_lengths_l631_63149

/-- Represents a right triangle with integer leg lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths possible for line segments
    drawn from a vertex to points on the hypotenuse -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

/-- The main theorem stating that for a right triangle with legs 24 and 25,
    there are exactly 16 distinct integer lengths possible -/
theorem sixteen_integer_lengths :
  ∃ (t : RightTriangle), t.de = 24 ∧ t.ef = 25 ∧ countIntegerLengths t = 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_integer_lengths_l631_63149


namespace NUMINAMATH_CALUDE_smallest_candy_count_l631_63124

theorem smallest_candy_count : ∃ (n : ℕ), 
  n = 127 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < n → ¬((m + 6) % 7 = 0 ∧ (m - 7) % 4 = 0)) ∧
  (n + 6) % 7 = 0 ∧ 
  (n - 7) % 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l631_63124


namespace NUMINAMATH_CALUDE_correct_calculation_l631_63184

theorem correct_calculation (x : ℤ) (h : x + 26 = 61) : x + 62 = 97 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l631_63184


namespace NUMINAMATH_CALUDE_complex_equation_solution_l631_63170

theorem complex_equation_solution : 
  ∃ (z : ℂ), (5 : ℂ) + 2 * Complex.I * z = (1 : ℂ) - 6 * Complex.I * z ∧ z = Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l631_63170


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l631_63114

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Area of triangle ABC
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 15 ∧
  -- Relationship between b and c
  b - c = 2 ∧
  -- Given cosine of A
  Real.cos A = -(1/4) →
  -- Conclusions
  a = 8 ∧
  Real.sin C = Real.sqrt 15 / 8 ∧
  Real.cos (2 * A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16 := by
sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l631_63114


namespace NUMINAMATH_CALUDE_necklace_profit_calculation_l631_63102

/-- Calculates the profit for a single necklace. -/
def profit_per_necklace (charm_count : ℕ) (charm_cost : ℕ) (selling_price : ℕ) : ℕ :=
  selling_price - charm_count * charm_cost

/-- Calculates the total profit for a specific necklace type. -/
def total_profit_for_type (necklace_count : ℕ) (charm_count : ℕ) (charm_cost : ℕ) (selling_price : ℕ) : ℕ :=
  necklace_count * profit_per_necklace charm_count charm_cost selling_price

theorem necklace_profit_calculation :
  let type_a_profit := total_profit_for_type 45 8 10 125
  let type_b_profit := total_profit_for_type 35 12 18 280
  let type_c_profit := total_profit_for_type 25 15 12 350
  type_a_profit + type_b_profit + type_c_profit = 8515 := by
  sorry

end NUMINAMATH_CALUDE_necklace_profit_calculation_l631_63102


namespace NUMINAMATH_CALUDE_champagne_discount_percentage_l631_63142

-- Define the problem parameters
def hot_tub_capacity : ℝ := 40
def bottle_capacity : ℝ := 1
def quarts_per_gallon : ℝ := 4
def original_price_per_bottle : ℝ := 50
def total_spent_after_discount : ℝ := 6400

-- Define the theorem
theorem champagne_discount_percentage :
  let total_quarts : ℝ := hot_tub_capacity * quarts_per_gallon
  let total_bottles : ℝ := total_quarts / bottle_capacity
  let full_price : ℝ := total_bottles * original_price_per_bottle
  let discount_amount : ℝ := full_price - total_spent_after_discount
  let discount_percentage : ℝ := (discount_amount / full_price) * 100
  discount_percentage = 20 := by
  sorry


end NUMINAMATH_CALUDE_champagne_discount_percentage_l631_63142


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l631_63189

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l631_63189


namespace NUMINAMATH_CALUDE_sum_10_with_7_dice_l631_63133

/-- The number of ways to roll a sum of 10 with 7 fair 6-sided dice -/
def ways_to_roll_10_with_7_dice : ℕ :=
  Nat.choose 9 6

/-- The probability of rolling a sum of 10 with 7 fair 6-sided dice -/
def prob_sum_10_7_dice : ℚ :=
  ways_to_roll_10_with_7_dice / (6^7 : ℚ)

theorem sum_10_with_7_dice :
  ways_to_roll_10_with_7_dice = 84 ∧
  prob_sum_10_7_dice = 84 / (6^7 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sum_10_with_7_dice_l631_63133


namespace NUMINAMATH_CALUDE_largest_product_bound_l631_63136

theorem largest_product_bound (a : Fin 1985 → Fin 1985) (h : Function.Bijective a) :
  (Finset.range 1985).sup (λ k => (k + 1) * a (k + 1)) ≥ 993^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_product_bound_l631_63136


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l631_63169

theorem polar_coordinates_of_point (x y : ℝ) (h : (x, y) = (-2, -2 * Real.sqrt 3)) :
  ∃ (ρ θ : ℝ), ρ = 4 ∧ θ = (4 * π) / 3 ∧
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l631_63169


namespace NUMINAMATH_CALUDE_power_two_2017_mod_7_l631_63176

theorem power_two_2017_mod_7 : 2^2017 % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_2017_mod_7_l631_63176


namespace NUMINAMATH_CALUDE_remaining_children_fed_theorem_l631_63152

/-- Represents the capacity of a meal in terms of adults and children -/
structure MealCapacity where
  adults : ℕ
  children : ℕ

/-- Calculates the number of children that can be fed with the remaining food -/
def remainingChildrenFed (capacity : MealCapacity) (adultsEaten : ℕ) : ℕ :=
  let remainingAdults := capacity.adults - adultsEaten
  (remainingAdults * capacity.children) / capacity.adults

/-- Theorem stating that given a meal for 70 adults or 90 children, 
    if 42 adults have eaten, the remaining food can feed 36 children -/
theorem remaining_children_fed_theorem (capacity : MealCapacity) 
  (h1 : capacity.adults = 70)
  (h2 : capacity.children = 90)
  (h3 : adultsEaten = 42) :
  remainingChildrenFed capacity adultsEaten = 36 := by
  sorry

#eval remainingChildrenFed { adults := 70, children := 90 } 42

end NUMINAMATH_CALUDE_remaining_children_fed_theorem_l631_63152


namespace NUMINAMATH_CALUDE_diplomats_not_speaking_russian_l631_63111

theorem diplomats_not_speaking_russian (total : ℕ) (french : ℕ) (neither_percent : ℚ) (both_percent : ℚ) :
  total = 150 →
  french = 17 →
  neither_percent = 1/5 →
  both_percent = 1/10 →
  ∃ (not_russian : ℕ), not_russian = 32 :=
by sorry

end NUMINAMATH_CALUDE_diplomats_not_speaking_russian_l631_63111


namespace NUMINAMATH_CALUDE_exponent_calculations_l631_63158

theorem exponent_calculations (a : ℝ) (h : a ≠ 0) : 
  (a^3 + a^3 ≠ a^6) ∧ 
  ((a^2)^3 ≠ a^5) ∧ 
  (a^2 * a^4 ≠ a^8) ∧ 
  (a^4 / a^3 = a) := by sorry

end NUMINAMATH_CALUDE_exponent_calculations_l631_63158


namespace NUMINAMATH_CALUDE_drum_size_correct_l631_63105

/-- Represents the size of the drum in gallons -/
def D : ℝ := 54.99

/-- Represents the amount of 100% antifreeze used in gallons -/
def pure_antifreeze : ℝ := 6.11

/-- Represents the percentage of antifreeze in the final mixture -/
def final_mixture_percent : ℝ := 0.20

/-- Represents the percentage of antifreeze in the initial diluted mixture -/
def initial_diluted_percent : ℝ := 0.10

/-- Theorem stating that the given conditions result in the correct drum size -/
theorem drum_size_correct : 
  pure_antifreeze + (D - pure_antifreeze) * initial_diluted_percent = D * final_mixture_percent := by
  sorry

#check drum_size_correct

end NUMINAMATH_CALUDE_drum_size_correct_l631_63105


namespace NUMINAMATH_CALUDE_tan_graph_property_l631_63163

theorem tan_graph_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, x ≠ π / 4 → ∃ y, y = a * Real.tan (b * x)) →
  (3 = a * Real.tan (b * π / 8)) →
  ab = 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_graph_property_l631_63163


namespace NUMINAMATH_CALUDE_probability_of_white_ball_is_five_eighths_l631_63127

/-- Represents the color of a ball -/
inductive Color
| White
| NonWhite

/-- Represents a bag of balls -/
def Bag := List Color

/-- The number of balls initially in the bag -/
def initialBallCount : Nat := 3

/-- Generates all possible initial configurations of the bag -/
def allPossibleInitialBags : List Bag :=
  sorry

/-- Adds a white ball to a bag -/
def addWhiteBall (bag : Bag) : Bag :=
  sorry

/-- Calculates the probability of drawing a white ball from a bag -/
def probabilityOfWhite (bag : Bag) : Rat :=
  sorry

/-- Calculates the average probability across all possible scenarios -/
def averageProbability (bags : List Bag) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem probability_of_white_ball_is_five_eighths :
  averageProbability (allPossibleInitialBags.map addWhiteBall) = 5/8 :=
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_is_five_eighths_l631_63127


namespace NUMINAMATH_CALUDE_pq_length_l631_63128

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  -- PQR is a right-angled triangle
  (Q.1 - P.1) * (R.2 - P.2) = (R.1 - P.1) * (Q.2 - P.2) ∧
  -- Angle PQR is 45°
  (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) * Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) / Real.sqrt 2 ∧
  -- PR = 10
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = 100

-- Theorem statement
theorem pq_length (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_pq_length_l631_63128


namespace NUMINAMATH_CALUDE_vector_simplification_l631_63166

variable {V : Type*} [AddCommGroup V]

variable (A B C M O : V)

theorem vector_simplification :
  (B - A) + (B - M) + (O - B) + (C - B) + (M - O) = C - A :=
by sorry

end NUMINAMATH_CALUDE_vector_simplification_l631_63166
