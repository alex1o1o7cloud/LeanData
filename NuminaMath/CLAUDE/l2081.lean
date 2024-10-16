import Mathlib

namespace NUMINAMATH_CALUDE_club_truncator_probability_l2081_208198

/-- The number of matches played by Club Truncator -/
def num_matches : ℕ := 8

/-- The probability of winning, losing, or tying a single match -/
def single_match_prob : ℚ := 1/3

/-- The probability of finishing with more wins than losses -/
def more_wins_prob : ℚ := 5483/13122

theorem club_truncator_probability :
  (num_matches = 8) →
  (single_match_prob = 1/3) →
  (more_wins_prob = 5483/13122) :=
by sorry

end NUMINAMATH_CALUDE_club_truncator_probability_l2081_208198


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l2081_208130

/-- Calculates overtime hours given regular pay rate, regular hours, and total pay -/
def calculate_overtime_hours (regular_rate : ℚ) (regular_hours : ℚ) (total_pay : ℚ) : ℚ :=
  let regular_pay := regular_rate * regular_hours
  let overtime_rate := 2 * regular_rate
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate

/-- Theorem stating that given the problem conditions, the overtime hours are 11 -/
theorem overtime_hours_calculation :
  let regular_rate : ℚ := 3
  let regular_hours : ℚ := 40
  let total_pay : ℚ := 186
  calculate_overtime_hours regular_rate regular_hours total_pay = 11 := by
  sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l2081_208130


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2081_208169

theorem arithmetic_sequence_middle_term (a₁ a₃ z : ℤ) : 
  a₁ = 3^2 → a₃ = 3^4 → (a₃ - z = z - a₁) → z = 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2081_208169


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l2081_208107

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 1) (a - 2)

-- Statement of the theorem
theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → z a = Complex.I * Complex.im (z a)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ z a = Complex.I * Complex.im (z a)) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l2081_208107


namespace NUMINAMATH_CALUDE_coefficient_sum_l2081_208117

theorem coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = -1 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_l2081_208117


namespace NUMINAMATH_CALUDE_simultaneous_truth_probability_l2081_208113

/-- The probability of A telling the truth -/
def prob_A_truth : ℝ := 0.8

/-- The probability of B telling the truth -/
def prob_B_truth : ℝ := 0.6

/-- The probability of A and B telling the truth simultaneously -/
def prob_both_truth : ℝ := prob_A_truth * prob_B_truth

theorem simultaneous_truth_probability :
  prob_both_truth = 0.48 :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_truth_probability_l2081_208113


namespace NUMINAMATH_CALUDE_fourth_column_unique_l2081_208179

/-- Represents a 9x9 Sudoku grid -/
def SudokuGrid := Fin 9 → Fin 9 → Fin 9

/-- Checks if a number is valid in a given position -/
def isValid (grid : SudokuGrid) (row col num : Fin 9) : Prop :=
  (∀ i : Fin 9, grid i col ≠ num) ∧
  (∀ j : Fin 9, grid row j ≠ num) ∧
  (∀ i j : Fin 3, grid (3 * (row / 3) + i) (3 * (col / 3) + j) ≠ num)

/-- Checks if the entire grid is valid -/
def isValidGrid (grid : SudokuGrid) : Prop :=
  ∀ row col : Fin 9, isValid grid row col (grid row col)

/-- Represents the pre-filled numbers in the 4th column -/
def fourthColumnPrefilled : Fin 9 → Option (Fin 9)
  | 0 => some 3
  | 1 => some 2
  | 3 => some 4
  | 7 => some 5
  | 8 => some 1
  | _ => none

/-- The theorem to be proved -/
theorem fourth_column_unique (grid : SudokuGrid) :
  isValidGrid grid →
  (∀ row : Fin 9, (fourthColumnPrefilled row).map (grid row 3) = fourthColumnPrefilled row) →
  (∀ row : Fin 9, grid row 3 = match row with
    | 0 => 3 | 1 => 2 | 2 => 7 | 3 => 4 | 4 => 6 | 5 => 8 | 6 => 9 | 7 => 5 | 8 => 1) :=
by sorry

end NUMINAMATH_CALUDE_fourth_column_unique_l2081_208179


namespace NUMINAMATH_CALUDE_max_a6_value_l2081_208158

theorem max_a6_value (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h_order : a₁ ≤ a₂ ∧ a₂ ≤ a₃ ∧ a₃ ≤ a₄ ∧ a₄ ≤ a₅ ∧ a₅ ≤ a₆)
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 10)
  (h_sq_dev : (a₁ - 1)^2 + (a₂ - 1)^2 + (a₃ - 1)^2 + (a₄ - 1)^2 + (a₅ - 1)^2 + (a₆ - 1)^2 = 6) :
  a₆ ≤ 10/3 := by
sorry

end NUMINAMATH_CALUDE_max_a6_value_l2081_208158


namespace NUMINAMATH_CALUDE_smallest_perimeter_is_nine_l2081_208128

/-- A triangle with consecutive integer side lengths where the smallest side is even. -/
structure ConsecutiveIntegerTriangle where
  a : ℕ
  is_even : Even a
  satisfies_triangle_inequality : a + (a + 1) > (a + 2) ∧ a + (a + 2) > (a + 1) ∧ (a + 1) + (a + 2) > a

/-- The perimeter of a ConsecutiveIntegerTriangle. -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ := t.a + (t.a + 1) + (t.a + 2)

/-- The smallest possible perimeter of a ConsecutiveIntegerTriangle is 9. -/
theorem smallest_perimeter_is_nine :
  ∀ t : ConsecutiveIntegerTriangle, perimeter t ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_is_nine_l2081_208128


namespace NUMINAMATH_CALUDE_min_tiles_for_floor_l2081_208152

-- Define the length and breadth of the floor in centimeters
def floor_length : ℚ := 1625 / 100
def floor_width : ℚ := 1275 / 100

-- Define the function to calculate the number of tiles
def num_tiles (length width : ℚ) : ℕ :=
  let gcd := (Nat.gcd (Nat.floor (length * 100)) (Nat.floor (width * 100))) / 100
  let tile_area := gcd * gcd
  let floor_area := length * width
  Nat.ceil (floor_area / tile_area)

-- Theorem statement
theorem min_tiles_for_floor : num_tiles floor_length floor_width = 3315 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_floor_l2081_208152


namespace NUMINAMATH_CALUDE_train_crossing_time_l2081_208159

/-- Proves that a train 100 meters long, traveling at 36 km/hr, takes 10 seconds to cross an electric pole -/
theorem train_crossing_time : 
  let train_length : ℝ := 100  -- Length of the train in meters
  let train_speed_kmh : ℝ := 36  -- Speed of the train in km/hr
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)  -- Speed in m/s
  let crossing_time : ℝ := train_length / train_speed_ms  -- Time to cross in seconds
  crossing_time = 10 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2081_208159


namespace NUMINAMATH_CALUDE_sin_593_degrees_l2081_208144

theorem sin_593_degrees (h : Real.sin (37 * π / 180) = 3 / 5) :
  Real.sin (593 * π / 180) = -(3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sin_593_degrees_l2081_208144


namespace NUMINAMATH_CALUDE_faster_person_time_l2081_208151

/-- The time it takes for the slower person to type the report, in minutes. -/
def slower_time : ℝ := 180

/-- The ratio of the faster person's typing speed to the slower person's typing speed. -/
def speed_ratio : ℝ := 4

/-- Theorem stating that the faster person will take 45 minutes to type the report. -/
theorem faster_person_time (report_length : ℝ) : 
  let slower_speed := report_length / slower_time
  let faster_speed := speed_ratio * slower_speed
  report_length / faster_speed = 45 := by sorry

end NUMINAMATH_CALUDE_faster_person_time_l2081_208151


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2081_208122

/-- The perimeter of a rhombus with diagonals 8 and 30 inches is 4√241 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * s = 4 * Real.sqrt 241 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2081_208122


namespace NUMINAMATH_CALUDE_bacteria_growth_proof_l2081_208112

/-- The number of quadrupling cycles in two minutes -/
def quadrupling_cycles : ℕ := 8

/-- The number of bacteria after two minutes -/
def final_bacteria_count : ℕ := 4194304

/-- The growth factor for each cycle -/
def growth_factor : ℕ := 4

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 64

theorem bacteria_growth_proof :
  initial_bacteria * growth_factor ^ quadrupling_cycles = final_bacteria_count :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_proof_l2081_208112


namespace NUMINAMATH_CALUDE_sequence_sum_l2081_208145

theorem sequence_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ + 13*x₇ = 3)
  (eq2 : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ + 15*x₇ = 15)
  (eq3 : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ + 17*x₇ = 85) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ + 19*x₇ = 213 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l2081_208145


namespace NUMINAMATH_CALUDE_julia_cakes_remaining_l2081_208104

/-- 
Given:
- Julia bakes one less than 5 cakes per day
- Julia bakes for 6 days
- Clifford eats one cake every other day

Prove that Julia has 21 cakes remaining after 6 days
-/
theorem julia_cakes_remaining (cakes_per_day : ℕ) (days : ℕ) (clifford_eats : ℕ) : 
  cakes_per_day = 5 - 1 → 
  days = 6 → 
  clifford_eats = days / 2 → 
  cakes_per_day * days - clifford_eats = 21 := by
sorry

end NUMINAMATH_CALUDE_julia_cakes_remaining_l2081_208104


namespace NUMINAMATH_CALUDE_line_circle_distance_sum_l2081_208193

-- Define the lines and circle
def line_l1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def line_l2 (x y a : ℝ) : Prop := 4 * x - 2 * y + a = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the distance sum condition
def distance_sum_condition (a : ℝ) : Prop :=
  ∀ x y : ℝ, circle_C x y →
    (|2*x - y + 1| / Real.sqrt 5 + |4*x - 2*y + a| / Real.sqrt 20 = 2 * Real.sqrt 5)

-- Theorem statement
theorem line_circle_distance_sum (a : ℝ) :
  distance_sum_condition a → (a = 10 ∨ a = -18) :=
sorry

end NUMINAMATH_CALUDE_line_circle_distance_sum_l2081_208193


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_35_l2081_208192

theorem largest_four_digit_divisible_by_35 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 35 = 0 → n ≤ 9985 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_35_l2081_208192


namespace NUMINAMATH_CALUDE_scientific_notation_of_billion_yuan_l2081_208177

def billion : ℝ := 1000000000

theorem scientific_notation_of_billion_yuan :
  let amount : ℝ := 2.175 * billion
  ∃ (a n : ℝ), amount = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = 9 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_billion_yuan_l2081_208177


namespace NUMINAMATH_CALUDE_vacation_cost_equality_l2081_208102

/-- Proves that t - d + s = 20 given the vacation cost conditions --/
theorem vacation_cost_equality (tom_paid dorothy_paid sammy_paid t d s : ℚ) :
  tom_paid = 150 →
  dorothy_paid = 160 →
  sammy_paid = 210 →
  let total := tom_paid + dorothy_paid + sammy_paid
  let per_person := total / 3
  t = per_person - tom_paid →
  d = per_person - dorothy_paid →
  s = sammy_paid - per_person →
  t - d + s = 20 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_equality_l2081_208102


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2081_208140

theorem complex_modulus_problem (z : ℂ) : (Complex.I^3 * z = 1 + Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2081_208140


namespace NUMINAMATH_CALUDE_jerry_insult_points_l2081_208194

/-- Represents the point system in Mrs. Carlton's class -/
structure PointSystem where
  interrupt_points : ℕ
  throw_points : ℕ
  office_threshold : ℕ

/-- Represents Jerry's behavior -/
structure JerryBehavior where
  interrupts : ℕ
  insults : ℕ
  throws : ℕ

/-- Calculates the points for insults given the point system and Jerry's behavior -/
def insult_points (ps : PointSystem) (jb : JerryBehavior) : ℕ :=
  (ps.office_threshold - (ps.interrupt_points * jb.interrupts + ps.throw_points * jb.throws)) / jb.insults

/-- Theorem stating that Jerry gets 10 points for insulting his classmates -/
theorem jerry_insult_points :
  let ps : PointSystem := { interrupt_points := 5, throw_points := 25, office_threshold := 100 }
  let jb : JerryBehavior := { interrupts := 2, insults := 4, throws := 2 }
  insult_points ps jb = 10 := by
  sorry

end NUMINAMATH_CALUDE_jerry_insult_points_l2081_208194


namespace NUMINAMATH_CALUDE_copy_pages_proof_l2081_208131

/-- Given a cost per page in cents and a budget in dollars, 
    calculates the maximum number of pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (budget_dollars : ℕ) : ℕ :=
  (budget_dollars * 100) / cost_per_page

/-- Proves that with a cost of 3 cents per page and a budget of $15,
    the maximum number of pages that can be copied is 500. -/
theorem copy_pages_proof :
  max_pages_copied 3 15 = 500 := by
  sorry

#eval max_pages_copied 3 15

end NUMINAMATH_CALUDE_copy_pages_proof_l2081_208131


namespace NUMINAMATH_CALUDE_value_of_a_l2081_208168

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3*x + 1

theorem value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2081_208168


namespace NUMINAMATH_CALUDE_john_works_five_days_week_l2081_208127

/-- Represents John's work schedule and patient count --/
structure DoctorSchedule where
  patients_hospital1 : ℕ
  patients_hospital2 : ℕ
  total_patients_year : ℕ
  weeks_per_year : ℕ

/-- Calculates the number of days John works per week --/
def days_per_week (s : DoctorSchedule) : ℚ :=
  s.total_patients_year / (s.weeks_per_year * (s.patients_hospital1 + s.patients_hospital2))

/-- Theorem stating that John works 5 days a week --/
theorem john_works_five_days_week (s : DoctorSchedule)
  (h1 : s.patients_hospital1 = 20)
  (h2 : s.patients_hospital2 = 24)
  (h3 : s.total_patients_year = 11000)
  (h4 : s.weeks_per_year = 50) :
  days_per_week s = 5 := by
  sorry

#eval days_per_week { patients_hospital1 := 20, patients_hospital2 := 24, total_patients_year := 11000, weeks_per_year := 50 }

end NUMINAMATH_CALUDE_john_works_five_days_week_l2081_208127


namespace NUMINAMATH_CALUDE_c_investment_is_10500_l2081_208134

/-- Calculates the investment of partner C given the investments of A and B, 
    the total profit, and A's share of the profit. -/
def calculate_c_investment (a_investment b_investment total_profit a_profit : ℚ) : ℚ :=
  (a_investment * total_profit / a_profit) - a_investment - b_investment

/-- Theorem stating that given the specified conditions, C's investment is 10500. -/
theorem c_investment_is_10500 :
  let a_investment : ℚ := 6300
  let b_investment : ℚ := 4200
  let total_profit : ℚ := 12700
  let a_profit : ℚ := 3810
  calculate_c_investment a_investment b_investment total_profit a_profit = 10500 := by
  sorry

#eval calculate_c_investment 6300 4200 12700 3810

end NUMINAMATH_CALUDE_c_investment_is_10500_l2081_208134


namespace NUMINAMATH_CALUDE_smallest_q_value_l2081_208173

def sum_of_range (n : ℕ) : ℕ := n * (n + 1) / 2

theorem smallest_q_value (p : ℕ) : 
  let initial_sum := sum_of_range 6
  let total_count := 6 + p + q
  let total_sum := initial_sum + 5 * p + 7 * q
  let mean := 5.3
  ∃ q : ℕ, q ≥ 0 ∧ (total_sum : ℝ) / total_count = mean ∧ 
    ∀ q' : ℕ, q' ≥ 0 → (initial_sum + 5 * p + 7 * q' : ℝ) / (6 + p + q') = mean → q ≤ q'
  := by sorry

end NUMINAMATH_CALUDE_smallest_q_value_l2081_208173


namespace NUMINAMATH_CALUDE_wheat_profit_percentage_l2081_208190

/-- Calculates the profit percentage for wheat mixture sales --/
theorem wheat_profit_percentage
  (weight1 : ℝ) (price1 : ℝ) (weight2 : ℝ) (price2 : ℝ) (selling_price : ℝ)
  (h1 : weight1 = 30)
  (h2 : price1 = 11.5)
  (h3 : weight2 = 20)
  (h4 : price2 = 14.25)
  (h5 : selling_price = 17.01) :
  let total_cost := weight1 * price1 + weight2 * price2
  let total_weight := weight1 + weight2
  let cost_per_kg := total_cost / total_weight
  let total_selling_price := selling_price * total_weight
  let profit := total_selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  ∃ ε > 0, abs (profit_percentage - 35) < ε :=
by sorry

end NUMINAMATH_CALUDE_wheat_profit_percentage_l2081_208190


namespace NUMINAMATH_CALUDE_max_cards_per_box_l2081_208170

/-- Given a total of 94 cards and 6 cards in an unfilled box, 
    prove that the maximum number of cards a full box can hold is 22. -/
theorem max_cards_per_box (total_cards : ℕ) (cards_in_unfilled_box : ℕ) 
  (h1 : total_cards = 94) (h2 : cards_in_unfilled_box = 6) :
  ∃ (max_cards_per_box : ℕ), 
    max_cards_per_box = 22 ∧ 
    max_cards_per_box > cards_in_unfilled_box ∧
    (total_cards - cards_in_unfilled_box) % max_cards_per_box = 0 ∧
    ∀ n : ℕ, n > max_cards_per_box → (total_cards - cards_in_unfilled_box) % n ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_cards_per_box_l2081_208170


namespace NUMINAMATH_CALUDE_percent_equality_l2081_208143

theorem percent_equality (x : ℝ) : (35 / 100 * 400 = 20 / 100 * x) → x = 700 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l2081_208143


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l2081_208101

/-- The area of a triangle formed by a point on a parabola, its focus, and the origin -/
theorem parabola_triangle_area :
  ∀ (x y : ℝ),
  y^2 = 8*x →                   -- Point (x, y) is on the parabola y² = 8x
  (x - 2)^2 + y^2 = 5^2 →       -- Distance from (x, y) to focus (2, 0) is 5
  (1/2) * 2 * y = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l2081_208101


namespace NUMINAMATH_CALUDE_sat_score_improvement_l2081_208118

theorem sat_score_improvement (first_score second_score : ℕ) 
  (h1 : first_score = 1000) 
  (h2 : second_score = 1100) : 
  (second_score - first_score) / first_score * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sat_score_improvement_l2081_208118


namespace NUMINAMATH_CALUDE_descent_time_specific_garage_l2081_208110

/-- Represents a parking garage with specified characteristics -/
structure ParkingGarage where
  floors : ℕ
  gateInterval : ℕ
  gateTime : ℕ
  floorDistance : ℕ
  drivingSpeed : ℕ

/-- Calculates the total time to descend the parking garage -/
def descentTime (garage : ParkingGarage) : ℕ :=
  let drivingTime := (garage.floors - 1) * (garage.floorDistance / garage.drivingSpeed)
  let gateCount := (garage.floors - 1) / garage.gateInterval
  let gateTime := gateCount * garage.gateTime
  drivingTime + gateTime

/-- The theorem stating the total descent time for the specific garage -/
theorem descent_time_specific_garage :
  let garage : ParkingGarage := {
    floors := 12,
    gateInterval := 3,
    gateTime := 120,
    floorDistance := 800,
    drivingSpeed := 10
  }
  descentTime garage = 1240 := by sorry

end NUMINAMATH_CALUDE_descent_time_specific_garage_l2081_208110


namespace NUMINAMATH_CALUDE_gcd_of_24_and_36_l2081_208138

theorem gcd_of_24_and_36 : Nat.gcd 24 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_24_and_36_l2081_208138


namespace NUMINAMATH_CALUDE_large_cartridge_pages_large_cartridge_pages_proof_l2081_208189

theorem large_cartridge_pages : ℕ → ℕ → ℕ → Prop :=
  fun small_pages medium_pages large_pages =>
    (small_pages = 600) →
    (3 * small_pages = 2 * medium_pages) →
    (3 * medium_pages = 2 * large_pages) →
    (large_pages = 1350)

-- The proof would go here
theorem large_cartridge_pages_proof :
  ∃ (small_pages medium_pages large_pages : ℕ),
    large_cartridge_pages small_pages medium_pages large_pages :=
sorry

end NUMINAMATH_CALUDE_large_cartridge_pages_large_cartridge_pages_proof_l2081_208189


namespace NUMINAMATH_CALUDE_rain_thunder_prob_is_correct_l2081_208166

/-- The probability of rain with thunder on both Monday and Tuesday -/
def rain_thunder_prob : ℝ :=
  let rain_monday_prob : ℝ := 0.40
  let rain_tuesday_prob : ℝ := 0.30
  let thunder_given_rain_prob : ℝ := 0.10
  let rain_both_days_prob : ℝ := rain_monday_prob * rain_tuesday_prob
  let thunder_both_days_given_rain_prob : ℝ := thunder_given_rain_prob * thunder_given_rain_prob
  rain_both_days_prob * thunder_both_days_given_rain_prob * 100

theorem rain_thunder_prob_is_correct : rain_thunder_prob = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_rain_thunder_prob_is_correct_l2081_208166


namespace NUMINAMATH_CALUDE_parabola_inequality_l2081_208136

/-- A parabola with x = 1 as its axis of symmetry -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  axis_symmetry : -b / (2 * a) = 1

theorem parabola_inequality (p : Parabola) : 2 * p.c < 3 * p.b := by
  sorry

end NUMINAMATH_CALUDE_parabola_inequality_l2081_208136


namespace NUMINAMATH_CALUDE_sin_2A_value_l2081_208165

theorem sin_2A_value (A : Real) (h : Real.cos (π/4 + A) = 5/13) : 
  Real.sin (2 * A) = 119/169 := by
  sorry

end NUMINAMATH_CALUDE_sin_2A_value_l2081_208165


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2081_208199

theorem polynomial_divisibility (a b c d m : ℤ) 
  (h1 : (5 : ℤ) ∣ (a * m^3 + b * m^2 + c * m + d))
  (h2 : ¬((5 : ℤ) ∣ d)) :
  ∃ n : ℤ, (5 : ℤ) ∣ (d * n^3 + c * n^2 + b * n + a) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2081_208199


namespace NUMINAMATH_CALUDE_operations_equality_l2081_208137

theorem operations_equality : 3 * 5 + 7 * 9 = 78 := by
  sorry

end NUMINAMATH_CALUDE_operations_equality_l2081_208137


namespace NUMINAMATH_CALUDE_quadratic_point_value_l2081_208149

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_point_value 
  (a b c : ℝ) 
  (h1 : ∀ x, quadratic_function a b c x ≤ 4)
  (h2 : quadratic_function a b c 2 = 4)
  (h3 : quadratic_function a b c 0 = -7) :
  quadratic_function a b c 5 = -83/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_value_l2081_208149


namespace NUMINAMATH_CALUDE_problem_solution_l2081_208196

theorem problem_solution : ∃ x : ℝ, 10 * x = 2 * x - 36 ∧ x = -4.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2081_208196


namespace NUMINAMATH_CALUDE_sabrina_cookies_l2081_208119

theorem sabrina_cookies (initial_cookies : ℕ) (final_cookies : ℕ) 
  (h1 : initial_cookies = 20) 
  (h2 : final_cookies = 5) : ℕ :=
  let cookies_to_brother := 10
  let cookies_from_mother := cookies_to_brother / 2
  let total_before_sister := initial_cookies - cookies_to_brother + cookies_from_mother
  let cookies_kept := total_before_sister / 3
  by
    have h3 : cookies_kept = final_cookies := by sorry
    have h4 : total_before_sister = cookies_kept * 3 := by sorry
    have h5 : initial_cookies - cookies_to_brother + cookies_from_mother = total_before_sister := by sorry
    exact cookies_to_brother

end NUMINAMATH_CALUDE_sabrina_cookies_l2081_208119


namespace NUMINAMATH_CALUDE_rational_sum_product_equality_l2081_208111

theorem rational_sum_product_equality : ∃ (a b : ℚ), a ≠ b ∧ a + b = a * b ∧ a = 3/2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_product_equality_l2081_208111


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2081_208139

theorem solution_set_inequality (x : ℝ) : x / (x + 1) ≤ 0 ↔ x ∈ Set.Ioc (-1) 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2081_208139


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l2081_208164

theorem cubic_root_sum_product (p q r : ℂ) : 
  (2 * p^3 - 4 * p^2 + 7 * p - 3 = 0) →
  (2 * q^3 - 4 * q^2 + 7 * q - 3 = 0) →
  (2 * r^3 - 4 * r^2 + 7 * r - 3 = 0) →
  p * q + q * r + r * p = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l2081_208164


namespace NUMINAMATH_CALUDE_f_nonnegative_when_a_is_one_f_has_two_zeros_iff_a_in_open_unit_interval_l2081_208123

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1 - 2 * log x

theorem f_nonnegative_when_a_is_one (x : ℝ) (h : x > 0) :
  f 1 x ≥ 0 := by sorry

theorem f_has_two_zeros_iff_a_in_open_unit_interval (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔
  0 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_f_nonnegative_when_a_is_one_f_has_two_zeros_iff_a_in_open_unit_interval_l2081_208123


namespace NUMINAMATH_CALUDE_problem_solution_l2081_208148

theorem problem_solution (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2004 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2081_208148


namespace NUMINAMATH_CALUDE_set_operations_l2081_208100

def U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

theorem set_operations :
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((A ∩ B) ∩ C = ∅) ∧
  ((U \ A) ∩ (U \ B) = {0, 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2081_208100


namespace NUMINAMATH_CALUDE_largest_two_digit_with_digit_product_12_l2081_208114

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem largest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_with_digit_product_12_l2081_208114


namespace NUMINAMATH_CALUDE_erased_number_proof_l2081_208161

/-- The number of integers in the original sequence -/
def n : ℕ := 71

/-- The average of the remaining numbers after one is erased -/
def average : ℚ := 37 + 11/19

/-- The erased number -/
def x : ℕ := 2704

theorem erased_number_proof :
  (n * (n + 1) / 2 - x) / (n - 1) = average → x = 2704 := by
  sorry

end NUMINAMATH_CALUDE_erased_number_proof_l2081_208161


namespace NUMINAMATH_CALUDE_find_other_number_l2081_208174

theorem find_other_number (A B : ℕ+) (hA : A = 24) (hHCF : Nat.gcd A B = 16) (hLCM : Nat.lcm A B = 312) : B = 208 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l2081_208174


namespace NUMINAMATH_CALUDE_certain_number_theorem_l2081_208120

theorem certain_number_theorem (z : ℕ) (h1 : z > 0) (h2 : 4 ∣ z) :
  (∃ n : ℕ, (z * (2 + 4 + z) + n) % 2 = 1 ∧
   ∀ m : ℕ, (z * (2 + 4 + z) + m) % 2 = 1 → n ≤ m) →
  (z * (2 + 4 + z) + 1) % 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_theorem_l2081_208120


namespace NUMINAMATH_CALUDE_dislike_both_tv_and_sports_l2081_208176

def total_surveyed : ℕ := 1500
def tv_dislike_percentage : ℚ := 40 / 100
def sports_dislike_percentage : ℚ := 15 / 100

theorem dislike_both_tv_and_sports :
  ∃ (n : ℕ), n = (total_surveyed : ℚ) * tv_dislike_percentage * sports_dislike_percentage ∧ n = 90 :=
by sorry

end NUMINAMATH_CALUDE_dislike_both_tv_and_sports_l2081_208176


namespace NUMINAMATH_CALUDE_square_sum_difference_equals_338_l2081_208163

theorem square_sum_difference_equals_338 :
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 338 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_difference_equals_338_l2081_208163


namespace NUMINAMATH_CALUDE_angelina_driving_equation_l2081_208142

/-- Represents the driving scenario of Angelina --/
structure DrivingScenario where
  initial_speed : ℝ
  rest_time : ℝ
  final_speed : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The equation for Angelina's driving time before rest --/
def driving_equation (s : DrivingScenario) (t : ℝ) : Prop :=
  s.initial_speed * t + s.final_speed * (s.total_time - s.rest_time / 60 - t) = s.total_distance

/-- Theorem stating that the given equation correctly represents Angelina's driving scenario --/
theorem angelina_driving_equation :
  ∃ (s : DrivingScenario),
    s.initial_speed = 60 ∧
    s.rest_time = 15 ∧
    s.final_speed = 90 ∧
    s.total_distance = 255 ∧
    s.total_time = 4 ∧
    ∀ (t : ℝ), driving_equation s t ↔ (60 * t + 90 * (15 / 4 - t) = 255) :=
  sorry

end NUMINAMATH_CALUDE_angelina_driving_equation_l2081_208142


namespace NUMINAMATH_CALUDE_nested_sqrt_bounds_l2081_208181

theorem nested_sqrt_bounds (x : ℝ) (h : x = Real.sqrt (3 + x)) : 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_bounds_l2081_208181


namespace NUMINAMATH_CALUDE_symmetric_difference_A_B_l2081_208121

def A : Set ℤ := {1, 2}
def B : Set ℤ := {x : ℤ | |x| < 2}

def set_difference (X Y : Set ℤ) : Set ℤ := {x : ℤ | x ∈ X ∧ x ∉ Y}
def symmetric_difference (X Y : Set ℤ) : Set ℤ := (set_difference X Y) ∪ (set_difference Y X)

theorem symmetric_difference_A_B :
  symmetric_difference A B = {-1, 0, 2} := by
  sorry

end NUMINAMATH_CALUDE_symmetric_difference_A_B_l2081_208121


namespace NUMINAMATH_CALUDE_price_change_l2081_208154

theorem price_change (q r : ℝ) (original_price : ℝ) :
  (original_price * (1 + q / 100) * (1 - r / 100) = 1) →
  (original_price = 1 / ((1 + q / 100) * (1 - r / 100))) :=
by sorry

end NUMINAMATH_CALUDE_price_change_l2081_208154


namespace NUMINAMATH_CALUDE_cubic_roots_existence_l2081_208197

theorem cubic_roots_existence (a b c : ℝ) : 
  (a + b + c = 6 ∧ a * b + b * c + c * a = 9) →
  (¬ (a^4 + b^4 + c^4 = 260) ∧ ∃ (x y z : ℝ), x + y + z = 6 ∧ x * y + y * z + z * x = 9 ∧ x^4 + y^4 + z^4 = 210) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_existence_l2081_208197


namespace NUMINAMATH_CALUDE_mark_can_bench_press_55_pounds_l2081_208115

/-- The weight that Mark can bench press -/
def marks_bench_press (daves_weight : ℝ) : ℝ :=
  let daves_bench_press := 3 * daves_weight
  let craigs_bench_press := 0.2 * daves_bench_press
  craigs_bench_press - 50

/-- Proof that Mark can bench press 55 pounds -/
theorem mark_can_bench_press_55_pounds :
  marks_bench_press 175 = 55 := by
  sorry

end NUMINAMATH_CALUDE_mark_can_bench_press_55_pounds_l2081_208115


namespace NUMINAMATH_CALUDE_quartic_polynomial_root_relation_l2081_208157

/-- Given a quartic polynomial ax^4 + bx^3 + cx^2 + dx + e = 0 with roots 4, -3, and 1, prove that (b+d)/a = -9/150 -/
theorem quartic_polynomial_root_relation (a b c d e : ℝ) (h1 : a ≠ 0) 
  (h2 : a * 4^4 + b * 4^3 + c * 4^2 + d * 4 + e = 0)
  (h3 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h4 : a * 1^4 + b * 1^3 + c * 1^2 + d * 1 + e = 0) : 
  (b + d) / a = -9 / 150 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_root_relation_l2081_208157


namespace NUMINAMATH_CALUDE_star_properties_l2081_208195

-- Define the star operation
def star (a b : ℝ) : ℝ := a + b + a * b

-- State the theorem
theorem star_properties :
  -- There exists an identity element E
  ∃ E : ℝ, (∀ a : ℝ, star a E = a) ∧ (star E E = E) ∧
  -- The operation is commutative
  (∀ a b : ℝ, star a b = star b a) ∧
  -- The operation is associative
  (∀ a b c : ℝ, star (star a b) c = star a (star b c)) :=
sorry

end NUMINAMATH_CALUDE_star_properties_l2081_208195


namespace NUMINAMATH_CALUDE_carton_height_is_70_l2081_208153

/-- Calculates the height of a carton given its base dimensions, soap box dimensions, and maximum capacity. -/
def carton_height (carton_length carton_width : ℕ) (box_length box_width box_height : ℕ) (max_boxes : ℕ) : ℕ :=
  let boxes_per_layer := (carton_length / box_length) * (carton_width / box_width)
  let num_layers := max_boxes / boxes_per_layer
  num_layers * box_height

/-- Theorem stating that the height of the carton is 70 inches given the specified conditions. -/
theorem carton_height_is_70 :
  carton_height 25 42 7 6 10 150 = 70 := by
  sorry

end NUMINAMATH_CALUDE_carton_height_is_70_l2081_208153


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2081_208191

theorem fraction_subtraction : (16 : ℚ) / 40 - (3 : ℚ) / 9 = (1 : ℚ) / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2081_208191


namespace NUMINAMATH_CALUDE_john_boxes_l2081_208167

theorem john_boxes (stan_boxes : ℕ) (joseph_percent : ℚ) (jules_more : ℕ) (john_percent : ℚ)
  (h1 : stan_boxes = 100)
  (h2 : joseph_percent = 80/100)
  (h3 : jules_more = 5)
  (h4 : john_percent = 20/100) :
  let joseph_boxes := stan_boxes * (1 - joseph_percent)
  let jules_boxes := joseph_boxes + jules_more
  let john_boxes := jules_boxes * (1 + john_percent)
  john_boxes = 30 := by sorry

end NUMINAMATH_CALUDE_john_boxes_l2081_208167


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2081_208108

theorem point_in_fourth_quadrant (x y : ℝ) : 
  (1 - Complex.I) * x = 1 + y * Complex.I → x > 0 ∧ y < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2081_208108


namespace NUMINAMATH_CALUDE_scientific_notation_of_14900_l2081_208150

theorem scientific_notation_of_14900 : 
  14900 = 1.49 * (10 : ℝ)^4 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_14900_l2081_208150


namespace NUMINAMATH_CALUDE_walkway_area_is_296_l2081_208105

/-- Represents the garden layout --/
structure Garden where
  rows : Nat
  columns : Nat
  bed_width : Nat
  bed_height : Nat
  walkway_width : Nat

/-- Calculates the total area of walkways in the garden --/
def walkway_area (g : Garden) : Nat :=
  let total_width := g.columns * g.bed_width + (g.columns + 1) * g.walkway_width
  let total_height := g.rows * g.bed_height + (g.rows + 1) * g.walkway_width
  let total_area := total_width * total_height
  let bed_area := g.rows * g.columns * g.bed_width * g.bed_height
  total_area - bed_area

/-- The theorem to be proved --/
theorem walkway_area_is_296 (g : Garden) :
  g.rows = 4 ∧ g.columns = 3 ∧ g.bed_width = 4 ∧ g.bed_height = 3 ∧ g.walkway_width = 2 →
  walkway_area g = 296 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_is_296_l2081_208105


namespace NUMINAMATH_CALUDE_jane_rejection_rate_l2081_208160

theorem jane_rejection_rate 
  (total_rejection_rate : ℝ) 
  (john_rejection_rate : ℝ) 
  (jane_inspection_fraction : ℝ) 
  (h1 : total_rejection_rate = 0.0075) 
  (h2 : john_rejection_rate = 0.005) 
  (h3 : jane_inspection_fraction = 0.8333333333333333) :
  let john_inspection_fraction := 1 - jane_inspection_fraction
  let jane_rejection_rate := (total_rejection_rate - john_rejection_rate * john_inspection_fraction) / jane_inspection_fraction
  jane_rejection_rate = 0.008 := by
sorry

end NUMINAMATH_CALUDE_jane_rejection_rate_l2081_208160


namespace NUMINAMATH_CALUDE_sniper_B_wins_l2081_208162

/-- Represents a sniper with probabilities of scoring 1, 2, and 3 points -/
structure Sniper where
  prob1 : ℝ
  prob2 : ℝ
  prob3 : ℝ

/-- Calculate the expected score for a sniper -/
def expectedScore (s : Sniper) : ℝ := 1 * s.prob1 + 2 * s.prob2 + 3 * s.prob3

/-- Sniper A with given probabilities -/
def sniperA : Sniper := { prob1 := 0.4, prob2 := 0.1, prob3 := 0.5 }

/-- Sniper B with given probabilities -/
def sniperB : Sniper := { prob1 := 0.1, prob2 := 0.6, prob3 := 0.3 }

/-- Theorem stating that Sniper B has a higher expected score than Sniper A -/
theorem sniper_B_wins : expectedScore sniperB > expectedScore sniperA := by
  sorry

end NUMINAMATH_CALUDE_sniper_B_wins_l2081_208162


namespace NUMINAMATH_CALUDE_percentage_of_number_l2081_208124

theorem percentage_of_number (x : ℝ) (h : (1/4) * (1/3) * (2/5) * x = 16) : 
  (40/100) * x = 192 := by sorry

end NUMINAMATH_CALUDE_percentage_of_number_l2081_208124


namespace NUMINAMATH_CALUDE_ellipse_range_theorem_l2081_208172

theorem ellipse_range_theorem :
  ∀ x y : ℝ, x^2/4 + y^2 = 1 → -Real.sqrt 17 ≤ 2*x + y ∧ 2*x + y ≤ Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_range_theorem_l2081_208172


namespace NUMINAMATH_CALUDE_exists_empty_selection_l2081_208141

/-- Represents a chessboard with pieces -/
structure Chessboard (n : ℕ) :=
  (board : Fin (2*n) → Fin (2*n) → Bool)
  (piece_count : Nat)
  (piece_count_eq : piece_count = 3*n)

/-- Represents a selection of rows and columns -/
structure Selection (n : ℕ) :=
  (rows : Fin n → Fin (2*n))
  (cols : Fin n → Fin (2*n))

/-- Checks if a selection results in an empty n × n chessboard -/
def is_empty_selection (cb : Chessboard n) (sel : Selection n) : Prop :=
  ∀ i j, ¬(cb.board (sel.rows i) (sel.cols j))

/-- Main theorem: There exists a selection that results in an empty n × n chessboard -/
theorem exists_empty_selection (n : ℕ) (cb : Chessboard n) :
  ∃ (sel : Selection n), is_empty_selection cb sel :=
sorry

end NUMINAMATH_CALUDE_exists_empty_selection_l2081_208141


namespace NUMINAMATH_CALUDE_difference_of_squares_303_297_l2081_208129

theorem difference_of_squares_303_297 : 303^2 - 297^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_303_297_l2081_208129


namespace NUMINAMATH_CALUDE_scale_model_height_l2081_208146

/-- The scale ratio used for the model -/
def scale_ratio : ℚ := 1 / 20

/-- The actual height of the United States Capitol in feet -/
def actual_height : ℕ := 289

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

/-- The height of the scale model rounded to the nearest foot -/
def model_height : ℕ := (round_to_nearest ((actual_height : ℚ) / scale_ratio)).toNat

theorem scale_model_height :
  model_height = 14 := by sorry

end NUMINAMATH_CALUDE_scale_model_height_l2081_208146


namespace NUMINAMATH_CALUDE_smallest_m_is_170_l2081_208183

/-- The quadratic equation 10x^2 - mx + 660 = 0 has integral solutions -/
def has_integral_solutions (m : ℤ) : Prop :=
  ∃ x : ℤ, 10 * x^2 - m * x + 660 = 0

/-- 170 is a value of m for which the equation has integral solutions -/
axiom solution_exists : has_integral_solutions 170

/-- For any positive integer less than 170, the equation does not have integral solutions -/
axiom no_smaller_solution : ∀ k : ℤ, 0 < k → k < 170 → ¬(has_integral_solutions k)

theorem smallest_m_is_170 : 
  (∃ m : ℤ, 0 < m ∧ has_integral_solutions m) ∧ 
  (∀ k : ℤ, 0 < k ∧ has_integral_solutions k → 170 ≤ k) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_170_l2081_208183


namespace NUMINAMATH_CALUDE_bOverA_equals_one_l2081_208180

theorem bOverA_equals_one (A B : ℤ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 →
    A / (x + 3) + B / (x^2 - 3*x) = (x^3 - 3*x^2 + 15*x - 9) / (x^3 + x^2 - 9*x)) →
  (B : ℚ) / A = 1 := by
sorry

end NUMINAMATH_CALUDE_bOverA_equals_one_l2081_208180


namespace NUMINAMATH_CALUDE_fishing_rod_price_theorem_l2081_208184

theorem fishing_rod_price_theorem :
  ∃ (a b c d : ℕ),
    -- Four-digit number condition
    1000 ≤ a * 1000 + b * 100 + c * 10 + d ∧ a * 1000 + b * 100 + c * 10 + d < 10000 ∧
    -- Digit relationships
    a = c + 1 ∧ a = d - 1 ∧
    -- Sum of digits
    a + b + c + d = 6 ∧
    -- Two-digit number difference
    10 * a + b = 10 * c + d + 7 ∧
    -- Product of ages
    a * 1000 + b * 100 + c * 10 + d = 61 * 3 * 11 :=
by sorry

end NUMINAMATH_CALUDE_fishing_rod_price_theorem_l2081_208184


namespace NUMINAMATH_CALUDE_value_of_u_minus_v_l2081_208109

theorem value_of_u_minus_v (u v : ℚ) 
  (eq1 : 5 * u - 6 * v = 31)
  (eq2 : 3 * u + 5 * v = 4) : 
  u - v = 5.3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_u_minus_v_l2081_208109


namespace NUMINAMATH_CALUDE_impossibility_l2081_208187

/-- The number of piles -/
def n : ℕ := 2018

/-- The i-th prime number -/
def p (i : ℕ) : ℕ := sorry

/-- The initial configuration of piles -/
def initial_config : Fin n → ℕ := λ i => p i.val

/-- The desired final configuration of piles -/
def final_config : Fin n → ℕ := λ _ => n

/-- Split operation: split a pile and add a chip to one of the new piles -/
def split (config : Fin n → ℕ) (i : Fin n) (k : ℕ) (add_to_first : Bool) : Fin n → ℕ := sorry

/-- Merge operation: merge two piles and add a chip to the merged pile -/
def merge (config : Fin n → ℕ) (i j : Fin n) : Fin n → ℕ := sorry

/-- Predicate to check if a configuration is reachable from the initial configuration -/
def is_reachable (config : Fin n → ℕ) : Prop := sorry

theorem impossibility : ¬ is_reachable final_config := by sorry

end NUMINAMATH_CALUDE_impossibility_l2081_208187


namespace NUMINAMATH_CALUDE_count_multiples_of_24_l2081_208185

def smallest_square_multiple_of_24 : ℕ := 144
def smallest_fourth_power_multiple_of_24 : ℕ := 1296

theorem count_multiples_of_24 :
  (Finset.range (smallest_fourth_power_multiple_of_24 / 24 + 1) ∩ 
   Finset.filter (λ n => n ≥ smallest_square_multiple_of_24 / 24) 
                 (Finset.range (smallest_fourth_power_multiple_of_24 / 24 + 1))).card = 49 :=
by sorry

end NUMINAMATH_CALUDE_count_multiples_of_24_l2081_208185


namespace NUMINAMATH_CALUDE_double_factorial_properties_l2081_208188

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def units_digit (n : ℕ) : ℕ := n % 10

theorem double_factorial_properties :
  (double_factorial 2011 * double_factorial 2010 = Nat.factorial 2011) ∧
  ¬(double_factorial 2010 = 2 * Nat.factorial 1005) ∧
  ¬(double_factorial 2010 * double_factorial 2010 = Nat.factorial 2011) ∧
  (units_digit (double_factorial 2011) = 5) := by
  sorry

end NUMINAMATH_CALUDE_double_factorial_properties_l2081_208188


namespace NUMINAMATH_CALUDE_cos_sin_pi_eighth_difference_l2081_208133

theorem cos_sin_pi_eighth_difference (π : Real) : 
  (Real.cos (π / 8))^4 - (Real.sin (π / 8))^4 = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_sin_pi_eighth_difference_l2081_208133


namespace NUMINAMATH_CALUDE_cosine_min_phase_l2081_208103

/-- Given a cosine function y = a cos(bx + c) where a, b, and c are positive constants,
    if the function reaches its first minimum at x = π/(2b), then c = π/2. -/
theorem cosine_min_phase (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x : ℝ, x ≥ 0 → a * Real.cos (b * x + c) ≥ a * Real.cos (b * (Real.pi / (2 * b)) + c)) →
  c = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_min_phase_l2081_208103


namespace NUMINAMATH_CALUDE_expression_evaluation_l2081_208125

theorem expression_evaluation : 
  2 * (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2081_208125


namespace NUMINAMATH_CALUDE_first_three_digit_in_square_sum_row_l2081_208175

/-- Represents a position in Pascal's triangle -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Returns the value at a given position in Pascal's triangle -/
def pascalValue (pos : Position) : Nat :=
  sorry

/-- Returns the sum of a row in Pascal's triangle -/
def rowSum (n : Nat) : Nat :=
  2^n

/-- Checks if a number is a three-digit number -/
def isThreeDigit (n : Nat) : Bool :=
  100 ≤ n ∧ n ≤ 999

/-- The theorem to be proved -/
theorem first_three_digit_in_square_sum_row :
  let pos := Position.mk 16 1
  (isThreeDigit (pascalValue pos)) ∧
  (∃ k : Nat, rowSum 16 = k * k) ∧
  (∀ n < 16, ∀ i ≤ n, ¬(isThreeDigit (pascalValue (Position.mk n i)) ∧ ∃ k : Nat, rowSum n = k * k)) :=
by sorry

end NUMINAMATH_CALUDE_first_three_digit_in_square_sum_row_l2081_208175


namespace NUMINAMATH_CALUDE_unique_k_for_lcm_l2081_208147

def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem unique_k_for_lcm : ∃! k : ℕ+, lcm (6^6) (9^9) k = 18^18 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_for_lcm_l2081_208147


namespace NUMINAMATH_CALUDE_equation_solution_l2081_208156

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = 3 + 2 * Real.sqrt 5 ∧ x₂ = 3 - 2 * Real.sqrt 5) ∧
    (∀ x : ℝ, 
      (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
       1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 4) ↔ 
      (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2081_208156


namespace NUMINAMATH_CALUDE_simplify_expression_solve_inequality_system_l2081_208135

-- Problem 1
theorem simplify_expression (x : ℝ) : (2*x + 1)^2 + x*(x - 4) = 5*x^2 + 1 := by
  sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) : 
  (3*x - 6 > 0 ∧ (5 - x)/2 < 1) ↔ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_solve_inequality_system_l2081_208135


namespace NUMINAMATH_CALUDE_smallest_number_l2081_208126

theorem smallest_number (a b c d : ℤ) 
  (ha : a = 2023) 
  (hb : b = 2022) 
  (hc : c = -2023) 
  (hd : d = -2022) : 
  c ≤ a ∧ c ≤ b ∧ c ≤ d := by
sorry

end NUMINAMATH_CALUDE_smallest_number_l2081_208126


namespace NUMINAMATH_CALUDE_present_value_is_490_l2081_208155

/-- Given a banker's discount and true discount, calculates the present value. -/
def present_value (bankers_discount : ℚ) (true_discount : ℚ) : ℚ :=
  true_discount^2 / (bankers_discount - true_discount)

/-- Theorem stating that for the given banker's discount and true discount, the present value is 490. -/
theorem present_value_is_490 :
  present_value 80 70 = 490 := by
  sorry

#eval present_value 80 70

end NUMINAMATH_CALUDE_present_value_is_490_l2081_208155


namespace NUMINAMATH_CALUDE_integer_set_property_l2081_208116

theorem integer_set_property (n : ℕ+) :
  ∃ (S : Finset ℤ), Finset.card S = n ∧
    ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b →
      ∃ (k : ℤ), a * b = k * (a - b)^2 :=
sorry

end NUMINAMATH_CALUDE_integer_set_property_l2081_208116


namespace NUMINAMATH_CALUDE_hit_at_least_once_and_not_hit_single_time_are_mutually_exclusive_and_complementary_l2081_208182

-- Define the sample space
def SampleSpace := Fin 3 → Bool

-- Define the event of hitting the target at least once
def HitAtLeastOnce (outcome : SampleSpace) : Prop :=
  ∃ i : Fin 3, outcome i = true

-- Define the event of not hitting the target a single time
def NotHitSingleTime (outcome : SampleSpace) : Prop :=
  ∀ i : Fin 3, outcome i = false

-- Theorem statement
theorem hit_at_least_once_and_not_hit_single_time_are_mutually_exclusive_and_complementary :
  (∀ outcome : SampleSpace, ¬(HitAtLeastOnce outcome ∧ NotHitSingleTime outcome)) ∧
  (∀ outcome : SampleSpace, HitAtLeastOnce outcome ↔ ¬NotHitSingleTime outcome) :=
sorry

end NUMINAMATH_CALUDE_hit_at_least_once_and_not_hit_single_time_are_mutually_exclusive_and_complementary_l2081_208182


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2081_208132

theorem quadratic_root_range (m : ℝ) (α β : ℝ) : 
  (∃ x, x^2 - m*x + 1 = 0) ∧ 
  (α^2 - m*α + 1 = 0) ∧ 
  (β^2 - m*β + 1 = 0) ∧ 
  (0 < α) ∧ (α < 1) ∧ 
  (1 < β) ∧ (β < 2) →
  (2 < m) ∧ (m < 5/2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2081_208132


namespace NUMINAMATH_CALUDE_cos_450_degrees_eq_zero_l2081_208178

theorem cos_450_degrees_eq_zero : Real.cos (450 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_450_degrees_eq_zero_l2081_208178


namespace NUMINAMATH_CALUDE_water_consumption_proof_l2081_208186

/-- Calculates the total water consumption for horses over a given period. -/
def total_water_consumption (initial_horses : ℕ) (added_horses : ℕ) (drinking_water : ℕ) (bathing_water : ℕ) (days : ℕ) : ℕ :=
  let total_horses := initial_horses + added_horses
  let daily_consumption_per_horse := drinking_water + bathing_water
  let daily_consumption := total_horses * daily_consumption_per_horse
  daily_consumption * days

/-- Proves that the total water consumption for the given conditions is 1568 liters. -/
theorem water_consumption_proof :
  total_water_consumption 3 5 5 2 28 = 1568 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_proof_l2081_208186


namespace NUMINAMATH_CALUDE_smallest_number_l2081_208171

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The number 85 in base 9 --/
def num1 : Nat := to_decimal [5, 8] 9

/-- The number 1000 in base 4 --/
def num2 : Nat := to_decimal [0, 0, 0, 1] 4

/-- The number 111111 in base 2 --/
def num3 : Nat := to_decimal [1, 1, 1, 1, 1, 1] 2

theorem smallest_number : num3 < num2 ∧ num3 < num1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2081_208171


namespace NUMINAMATH_CALUDE_base_six_addition_problem_l2081_208106

/-- Given a base-6 addition problem 5CD₆ + 52₆ = 64C₆, prove that C + D = 8 in base 10 -/
theorem base_six_addition_problem (C D : ℕ) : 
  (5 * 6^2 + C * 6 + D) + (5 * 6 + 2) = 6 * 6^2 + 4 * 6 + C →
  C + D = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_six_addition_problem_l2081_208106
