import Mathlib

namespace NUMINAMATH_CALUDE_messages_sent_theorem_l2741_274113

/-- Calculates the total number of messages sent over three days given the conditions -/
def totalMessages (luciaFirstDay : ℕ) (alinaDifference : ℕ) : ℕ :=
  let alinaFirstDay := luciaFirstDay - alinaDifference
  let firstDayTotal := luciaFirstDay + alinaFirstDay
  let luciaSecondDay := luciaFirstDay / 3
  let alinaSecondDay := alinaFirstDay * 2
  let secondDayTotal := luciaSecondDay + alinaSecondDay
  firstDayTotal + secondDayTotal + firstDayTotal

theorem messages_sent_theorem :
  totalMessages 120 20 = 680 := by
  sorry

#eval totalMessages 120 20

end NUMINAMATH_CALUDE_messages_sent_theorem_l2741_274113


namespace NUMINAMATH_CALUDE_first_coaster_speed_is_50_l2741_274118

/-- The speed of the first rollercoaster given the speeds of the other four and the average speed -/
def first_coaster_speed (second_speed third_speed fourth_speed fifth_speed average_speed : ℝ) : ℝ :=
  5 * average_speed - (second_speed + third_speed + fourth_speed + fifth_speed)

/-- Theorem stating that the first coaster's speed is 50 mph given the problem conditions -/
theorem first_coaster_speed_is_50 :
  first_coaster_speed 62 73 70 40 59 = 50 := by
  sorry

end NUMINAMATH_CALUDE_first_coaster_speed_is_50_l2741_274118


namespace NUMINAMATH_CALUDE_pizza_production_l2741_274130

theorem pizza_production (craig_day1 : ℕ) (craig_increase : ℕ) (heather_decrease : ℕ)
  (h1 : craig_day1 = 40)
  (h2 : craig_increase = 60)
  (h3 : heather_decrease = 20) :
  let heather_day1 := 4 * craig_day1
  let craig_day2 := craig_day1 + craig_increase
  let heather_day2 := craig_day2 - heather_decrease
  heather_day1 + craig_day1 + heather_day2 + craig_day2 = 380 := by
  sorry


end NUMINAMATH_CALUDE_pizza_production_l2741_274130


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2741_274100

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time interest : ℝ) (h1 : principal = 10000) 
  (h2 : time = 1) (h3 : interest = 500) : 
  (interest / (principal * time)) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l2741_274100


namespace NUMINAMATH_CALUDE_time_after_2011_minutes_l2741_274175

/-- Represents a time with day, hour, and minute components -/
structure DateTime where
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Converts total minutes to a DateTime structure -/
def minutesToDateTime (totalMinutes : ℕ) : DateTime :=
  let totalHours := totalMinutes / 60
  let days := totalHours / 24
  let hours := totalHours % 24
  let minutes := totalMinutes % 60
  { day := days + 1, hour := hours, minute := minutes }

/-- The starting date and time -/
def startDateTime : DateTime := { day := 1, hour := 0, minute := 0 }

/-- The number of minutes elapsed -/
def elapsedMinutes : ℕ := 2011

theorem time_after_2011_minutes :
  minutesToDateTime elapsedMinutes = { day := 2, hour := 9, minute := 31 } := by
  sorry

end NUMINAMATH_CALUDE_time_after_2011_minutes_l2741_274175


namespace NUMINAMATH_CALUDE_factorial_difference_is_cubic_polynomial_cubic_polynomial_form_l2741_274187

theorem factorial_difference_is_cubic_polynomial (n : ℕ) (h : n ≥ 9) :
  (((n + 3).factorial - (n + 2).factorial) / n.factorial : ℚ) = (n + 2)^2 * (n + 1) :=
by sorry

theorem cubic_polynomial_form (n : ℕ) (h : n ≥ 9) :
  ∃ (a b c d : ℚ), (n + 2)^2 * (n + 1) = a * n^3 + b * n^2 + c * n + d :=
by sorry

end NUMINAMATH_CALUDE_factorial_difference_is_cubic_polynomial_cubic_polynomial_form_l2741_274187


namespace NUMINAMATH_CALUDE_platform_length_l2741_274150

/-- Calculates the length of a platform given train specifications -/
theorem platform_length
  (train_length : ℝ)
  (time_tree : ℝ)
  (time_platform : ℝ)
  (h1 : train_length = 600)
  (h2 : time_tree = 60)
  (h3 : time_platform = 105) :
  let train_speed := train_length / time_tree
  let platform_length := train_speed * time_platform - train_length
  platform_length = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2741_274150


namespace NUMINAMATH_CALUDE_greatest_k_for_inequality_l2741_274199

theorem greatest_k_for_inequality : ∃! k : ℕ, 
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 → a * b * c = 1 → 
    (1 / a + 1 / b + 1 / c + k / (a + b + c + 1) ≥ 3 + k / 4)) ∧
  (∀ k' : ℕ, k' > k → 
    ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧
      1 / a + 1 / b + 1 / c + k' / (a + b + c + 1) < 3 + k' / 4) ∧
  k = 13 :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_for_inequality_l2741_274199


namespace NUMINAMATH_CALUDE_prism_base_side_length_l2741_274168

/-- Given a rectangular prism with a square base, prove that with the given dimensions and properties, the side length of the base is 2 meters. -/
theorem prism_base_side_length (height : ℝ) (density : ℝ) (weight : ℝ) (volume : ℝ) (side : ℝ) :
  height = 8 →
  density = 2700 →
  weight = 86400 →
  volume = weight / density →
  volume = side^2 * height →
  side = 2 := by
  sorry


end NUMINAMATH_CALUDE_prism_base_side_length_l2741_274168


namespace NUMINAMATH_CALUDE_circle_equation_tangent_line_l2741_274159

/-- The equation of a circle with center (3, -1) that is tangent to the line 3x + 4y = 0 is (x-3)² + (y+1)² = 1 -/
theorem circle_equation_tangent_line (x y : ℝ) : 
  let center : ℝ × ℝ := (3, -1)
  let line (x y : ℝ) := 3 * x + 4 * y = 0
  let circle_eq (x y : ℝ) := (x - center.1)^2 + (y - center.2)^2 = 1
  let is_tangent (circle : (ℝ → ℝ → Prop) ) (line : ℝ → ℝ → Prop) := 
    ∃ (x y : ℝ), circle x y ∧ line x y ∧ 
    ∀ (x' y' : ℝ), line x' y' → (x' = x ∧ y' = y) ∨ ¬(circle x' y')
  is_tangent circle_eq line → circle_eq x y := by
sorry


end NUMINAMATH_CALUDE_circle_equation_tangent_line_l2741_274159


namespace NUMINAMATH_CALUDE_product_four_consecutive_odd_integers_is_nine_l2741_274123

theorem product_four_consecutive_odd_integers_is_nine :
  ∃ n : ℤ, (2*n - 3) * (2*n - 1) * (2*n + 1) * (2*n + 3) = 9 :=
by sorry

end NUMINAMATH_CALUDE_product_four_consecutive_odd_integers_is_nine_l2741_274123


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l2741_274161

theorem square_perimeter_problem :
  ∀ (a b c : ℝ),
  (4 * a = 16) →  -- Perimeter of square A is 16
  (4 * b = 32) →  -- Perimeter of square B is 32
  (c = 4 * (b - a)) →  -- Side length of C is 4 times the difference of A and B's side lengths
  (4 * c = 64) :=  -- Perimeter of square C is 64
by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l2741_274161


namespace NUMINAMATH_CALUDE_product_equals_243_l2741_274186

theorem product_equals_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_243_l2741_274186


namespace NUMINAMATH_CALUDE_set_operations_l2741_274141

def A : Set ℕ := {x | x > 0 ∧ x < 11}
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6, 7}

theorem set_operations :
  (A ∩ C = {3, 4, 5, 6, 7}) ∧
  ((A \ B) = {5, 6, 7, 8, 9, 10}) ∧
  ((A \ (B ∪ C)) = {8, 9, 10}) ∧
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l2741_274141


namespace NUMINAMATH_CALUDE_largest_common_number_l2741_274139

theorem largest_common_number (n m : ℕ) : 
  67 = 1 + 6 * n ∧ 
  67 = 4 + 7 * m ∧ 
  67 ≤ 100 ∧ 
  ∀ k, (∃ p q : ℕ, k = 1 + 6 * p ∧ k = 4 + 7 * q ∧ k ≤ 100) → k ≤ 67 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_number_l2741_274139


namespace NUMINAMATH_CALUDE_quiz_statistics_l2741_274145

def scores : List ℕ := [7, 5, 6, 8, 7, 9]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem quiz_statistics :
  mean scores = 7 ∧ mode scores = 7 := by
  sorry

end NUMINAMATH_CALUDE_quiz_statistics_l2741_274145


namespace NUMINAMATH_CALUDE_traffic_police_distribution_l2741_274133

def officers : ℕ := 5
def specific_officers : ℕ := 2
def intersections : ℕ := 3

theorem traffic_police_distribution :
  (Nat.choose (officers - specific_officers + 1) (intersections - 1)) *
  (Nat.factorial intersections) = 36 := by sorry

end NUMINAMATH_CALUDE_traffic_police_distribution_l2741_274133


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2741_274180

def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2741_274180


namespace NUMINAMATH_CALUDE_order_of_abc_l2741_274172

noncomputable def a : ℝ := (4 - Real.log 4) / Real.exp 2
noncomputable def b : ℝ := Real.log 2 / 2
noncomputable def c : ℝ := 1 / Real.exp 1

theorem order_of_abc : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l2741_274172


namespace NUMINAMATH_CALUDE_candidate_d_votes_l2741_274122

theorem candidate_d_votes (total_votes : ℕ) (invalid_percentage : ℚ)
  (candidate_a_percentage : ℚ) (candidate_b_percentage : ℚ) (candidate_c_percentage : ℚ)
  (h1 : total_votes = 10000)
  (h2 : invalid_percentage = 1/4)
  (h3 : candidate_a_percentage = 2/5)
  (h4 : candidate_b_percentage = 3/10)
  (h5 : candidate_c_percentage = 1/5) :
  ↑total_votes * (1 - invalid_percentage) * (1 - (candidate_a_percentage + candidate_b_percentage + candidate_c_percentage)) = 750 := by
  sorry

#check candidate_d_votes

end NUMINAMATH_CALUDE_candidate_d_votes_l2741_274122


namespace NUMINAMATH_CALUDE_triangle_angle_C_l2741_274124

/-- Given a triangle ABC with side lengths a and c, and angle A, prove that C is either 60° or 120°. -/
theorem triangle_angle_C (a c : ℝ) (A : Real) (h1 : a = 2) (h2 : c = Real.sqrt 6) (h3 : A = π / 4) :
  let C := Real.arcsin ((c * Real.sin A) / a)
  C = π / 3 ∨ C = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l2741_274124


namespace NUMINAMATH_CALUDE_expression_evaluation_l2741_274181

theorem expression_evaluation (x y : ℚ) (hx : x = 1) (hy : y = -3) :
  ((x - 2*y)^2 + (3*x - y)*(3*x + y) - 3*y^2) / (-2*x) = -11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2741_274181


namespace NUMINAMATH_CALUDE_handshake_theorem_l2741_274132

theorem handshake_theorem (n : ℕ) (h : n = 30) :
  let total_handshakes := n * 3 / 2
  total_handshakes = 45 := by
  sorry

end NUMINAMATH_CALUDE_handshake_theorem_l2741_274132


namespace NUMINAMATH_CALUDE_tom_remaining_pieces_l2741_274160

/-- 
Given the initial number of boxes, the number of boxes given away, 
and the number of pieces per box, calculate the number of pieces Tom still had.
-/
def remaining_pieces (initial_boxes : ℕ) (boxes_given_away : ℕ) (pieces_per_box : ℕ) : ℕ :=
  (initial_boxes - boxes_given_away) * pieces_per_box

theorem tom_remaining_pieces : 
  remaining_pieces 12 7 6 = 30 := by
  sorry

#eval remaining_pieces 12 7 6

end NUMINAMATH_CALUDE_tom_remaining_pieces_l2741_274160


namespace NUMINAMATH_CALUDE_ones_digit_of_large_power_l2741_274195

theorem ones_digit_of_large_power : ∃ n : ℕ, n < 10 ∧ 34^(34*(17^17)) ≡ n [ZMOD 10] :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_large_power_l2741_274195


namespace NUMINAMATH_CALUDE_susan_initial_money_l2741_274198

theorem susan_initial_money (S : ℝ) : 
  S - (S / 5 + S / 4 + 120) = 1200 → S = 2400 := by
  sorry

end NUMINAMATH_CALUDE_susan_initial_money_l2741_274198


namespace NUMINAMATH_CALUDE_high_school_twelve_games_l2741_274143

/-- The number of teams in the "High School Twelve" basketball league -/
def num_teams : ℕ := 12

/-- The number of times each team plays every other team in the league -/
def games_per_pair : ℕ := 3

/-- The number of games each team plays against non-league teams -/
def non_league_games_per_team : ℕ := 6

/-- The total number of games in a season for the "High School Twelve" basketball league -/
def total_games : ℕ := (num_teams.choose 2) * games_per_pair + num_teams * non_league_games_per_team

theorem high_school_twelve_games :
  total_games = 270 := by sorry

end NUMINAMATH_CALUDE_high_school_twelve_games_l2741_274143


namespace NUMINAMATH_CALUDE_students_in_either_not_both_l2741_274142

/-- The number of students taking both geometry and statistics -/
def both_subjects : ℕ := 18

/-- The total number of students taking geometry -/
def geometry_total : ℕ := 35

/-- The number of students taking only statistics -/
def only_statistics : ℕ := 16

/-- Theorem: The number of students taking geometry or statistics but not both is 33 -/
theorem students_in_either_not_both : 
  (geometry_total - both_subjects) + only_statistics = 33 := by
  sorry

end NUMINAMATH_CALUDE_students_in_either_not_both_l2741_274142


namespace NUMINAMATH_CALUDE_negative_integer_equation_solution_l2741_274163

theorem negative_integer_equation_solution :
  ∃! (N : ℤ), N < 0 ∧ N + 2 * N^2 = 12 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_negative_integer_equation_solution_l2741_274163


namespace NUMINAMATH_CALUDE_expression_equals_one_l2741_274196

theorem expression_equals_one : 
  (144^2 - 12^2) / (120^2 - 18^2) * ((120-18)*(120+18)) / ((144-12)*(144+12)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2741_274196


namespace NUMINAMATH_CALUDE_complex_multiplication_l2741_274134

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (-1 + i) * (2 - i) = -1 + 3*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2741_274134


namespace NUMINAMATH_CALUDE_david_average_speed_l2741_274119

def distance : ℚ := 49 / 3  -- 16 1/3 miles as a fraction

def time : ℚ := 7 / 3  -- 2 hours and 20 minutes as a fraction of hours

def average_speed (d t : ℚ) : ℚ := d / t

theorem david_average_speed :
  average_speed distance time = 7 := by sorry

end NUMINAMATH_CALUDE_david_average_speed_l2741_274119


namespace NUMINAMATH_CALUDE_q_expression_l2741_274116

/-- Given a function q(x) satisfying the equation
    q(x) + (x^6 + 4x^4 + 5x^3 + 12x) = (8x^4 + 26x^3 + 15x^2 + 26x + 3),
    prove that q(x) = -x^6 + 4x^4 + 21x^3 + 15x^2 + 14x + 3 -/
theorem q_expression (q : ℝ → ℝ) 
    (h : ∀ x, q x + (x^6 + 4*x^4 + 5*x^3 + 12*x) = 8*x^4 + 26*x^3 + 15*x^2 + 26*x + 3) :
  ∀ x, q x = -x^6 + 4*x^4 + 21*x^3 + 15*x^2 + 14*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_q_expression_l2741_274116


namespace NUMINAMATH_CALUDE_simplify_expression_l2741_274109

theorem simplify_expression (x : ℝ) : ((-3 * x)^2) * (2 * x) = 18 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2741_274109


namespace NUMINAMATH_CALUDE_inequality_implication_l2741_274120

theorem inequality_implication (a b : ℝ) : 
  a^2 - b^2 + 2*a - 4*b - 3 ≠ 0 → a - b ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2741_274120


namespace NUMINAMATH_CALUDE_bus_rows_theorem_l2741_274110

/-- Represents a school bus with rows of seats split by an aisle -/
structure SchoolBus where
  total_students : ℕ
  students_per_section : ℕ
  sections_per_row : ℕ

/-- Calculates the number of rows in a school bus -/
def num_rows (bus : SchoolBus) : ℕ :=
  (bus.total_students / bus.students_per_section) / bus.sections_per_row

/-- Theorem stating that a bus with 52 students, 2 students per section, and 2 sections per row has 13 rows -/
theorem bus_rows_theorem (bus : SchoolBus) 
  (h1 : bus.total_students = 52)
  (h2 : bus.students_per_section = 2)
  (h3 : bus.sections_per_row = 2) :
  num_rows bus = 13 := by
  sorry

#eval num_rows { total_students := 52, students_per_section := 2, sections_per_row := 2 }

end NUMINAMATH_CALUDE_bus_rows_theorem_l2741_274110


namespace NUMINAMATH_CALUDE_mathematics_letter_probability_l2741_274156

theorem mathematics_letter_probability : 
  let alphabet_size : ℕ := 26
  let unique_letters : ℕ := 8
  (unique_letters : ℚ) / (alphabet_size : ℚ) = 4 / 13 :=
by sorry

end NUMINAMATH_CALUDE_mathematics_letter_probability_l2741_274156


namespace NUMINAMATH_CALUDE_fair_attendance_l2741_274148

/-- Proves the number of adults attending a fair given admission fees, total attendance, and total amount collected. -/
theorem fair_attendance 
  (child_fee : ℚ) 
  (adult_fee : ℚ) 
  (total_people : ℕ) 
  (total_amount : ℚ) 
  (h1 : child_fee = 3/2) 
  (h2 : adult_fee = 4) 
  (h3 : total_people = 2200) 
  (h4 : total_amount = 5050) : 
  ∃ (adults : ℕ), adults = 700 ∧ 
    ∃ (children : ℕ), 
      children + adults = total_people ∧ 
      child_fee * children + adult_fee * adults = total_amount := by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_l2741_274148


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l2741_274149

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 3) - 4
  f (-3) = -3 := by sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l2741_274149


namespace NUMINAMATH_CALUDE_no_four_identical_digits_in_1990_denominator_l2741_274131

theorem no_four_identical_digits_in_1990_denominator :
  ¬ ∃ (A : ℕ) (d : ℕ), 
    A > 0 ∧ A < 1990 ∧ d < 10 ∧
    ∃ (k : ℕ), (A * 10^k) % 1990 = d * 1111 :=
by sorry

end NUMINAMATH_CALUDE_no_four_identical_digits_in_1990_denominator_l2741_274131


namespace NUMINAMATH_CALUDE_sector_central_angle_l2741_274108

/-- Given a circular sector with perimeter 8 and area 3, 
    its central angle is either 6 or 2/3 radians. -/
theorem sector_central_angle (r l : ℝ) : 
  (2 * r + l = 8) →  -- perimeter condition
  (1 / 2 * l * r = 3) →  -- area condition
  (l / r = 6 ∨ l / r = 2 / 3) := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2741_274108


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l2741_274102

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (52 * x + 14) % 24 = 6 ∧
  ∀ (y : ℕ), y > 0 ∧ (52 * y + 14) % 24 = 6 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l2741_274102


namespace NUMINAMATH_CALUDE_squarePerimeter_doesnt_require_conditional_statements_only_squarePerimeter_doesnt_require_conditional_statements_l2741_274126

-- Define a type for the different problems
inductive Problem
  | oppositeNumber
  | squarePerimeter
  | maxOfThree
  | binaryToDecimal

-- Function to determine if a problem requires conditional statements
def requiresConditionalStatements (p : Problem) : Prop :=
  match p with
  | Problem.oppositeNumber => False
  | Problem.squarePerimeter => False
  | Problem.maxOfThree => True
  | Problem.binaryToDecimal => True

-- Theorem stating that the square perimeter problem doesn't require conditional statements
theorem squarePerimeter_doesnt_require_conditional_statements :
  ¬(requiresConditionalStatements Problem.squarePerimeter) :=
by
  sorry

-- Theorem stating that the square perimeter problem is the only one among the four that doesn't require conditional statements
theorem only_squarePerimeter_doesnt_require_conditional_statements :
  ∀ (p : Problem), ¬(requiresConditionalStatements p) → p = Problem.squarePerimeter :=
by
  sorry

end NUMINAMATH_CALUDE_squarePerimeter_doesnt_require_conditional_statements_only_squarePerimeter_doesnt_require_conditional_statements_l2741_274126


namespace NUMINAMATH_CALUDE_problem_statement_l2741_274144

def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → a^x < a^y

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem problem_statement (a : ℝ) (h1 : a > 0) (h2 : (p a ∨ q a) ∧ ¬(p a ∧ q a)) :
  a ≥ 4 ∨ (0 < a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l2741_274144


namespace NUMINAMATH_CALUDE_john_can_buy_max_notebooks_l2741_274167

/-- The amount of money John has, in cents -/
def johns_money : ℕ := 4575

/-- The cost of each notebook, in cents -/
def notebook_cost : ℕ := 325

/-- The maximum number of notebooks John can buy -/
def max_notebooks : ℕ := 14

theorem john_can_buy_max_notebooks :
  (max_notebooks * notebook_cost ≤ johns_money) ∧
  ∀ n : ℕ, n > max_notebooks → n * notebook_cost > johns_money :=
by sorry

end NUMINAMATH_CALUDE_john_can_buy_max_notebooks_l2741_274167


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2741_274105

theorem arithmetic_geometric_mean_inequality {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2741_274105


namespace NUMINAMATH_CALUDE_bottle_capacity_proof_l2741_274191

theorem bottle_capacity_proof (total_milk : ℝ) (bottle1_capacity : ℝ) (bottle1_milk : ℝ) (bottle2_capacity : ℝ) :
  total_milk = 8 →
  bottle1_capacity = 8 →
  bottle1_milk = 5.333333333333333 →
  (bottle1_milk / bottle1_capacity) = ((total_milk - bottle1_milk) / bottle2_capacity) →
  bottle2_capacity = 4 := by
sorry

end NUMINAMATH_CALUDE_bottle_capacity_proof_l2741_274191


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_185_l2741_274157

theorem modular_inverse_of_3_mod_185 :
  ∃ x : ℕ, x < 185 ∧ (3 * x) % 185 = 1 :=
by
  use 62
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_185_l2741_274157


namespace NUMINAMATH_CALUDE_scaling_circle_not_hyperbola_l2741_274137

-- Define a circle
def Circle := Set (ℝ × ℝ)

-- Define a scaling transformation
def ScalingTransformation := (ℝ × ℝ) → (ℝ × ℝ)

-- Define a hyperbola
def Hyperbola := Set (ℝ × ℝ)

-- Theorem statement
theorem scaling_circle_not_hyperbola (c : Circle) (s : ScalingTransformation) :
  ∀ h : Hyperbola, (s '' c) ≠ h :=
sorry

end NUMINAMATH_CALUDE_scaling_circle_not_hyperbola_l2741_274137


namespace NUMINAMATH_CALUDE_square_difference_factorization_l2741_274147

theorem square_difference_factorization (x y : ℝ) : 
  49 * x^2 - 36 * y^2 = (-6*y + 7*x) * (6*y + 7*x) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_factorization_l2741_274147


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l2741_274183

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p ∣ (36^2 + 49^2) ∧ 
    ∀ (q : ℕ), Nat.Prime q → q ∣ (36^2 + 49^2) → q ≤ p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l2741_274183


namespace NUMINAMATH_CALUDE_m_range_l2741_274146

-- Define propositions p and q
def p (x : ℝ) : Prop := x + 2 ≥ 0 ∧ x - 10 ≤ 0

def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬q x

-- State the theorem
theorem m_range (m : ℝ) :
  m > 0 ∧
  necessary_not_sufficient (p) (q m) →
  0 < m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l2741_274146


namespace NUMINAMATH_CALUDE_expression_equality_l2741_274190

theorem expression_equality (a b c n : ℝ) 
  (h1 : a + b = c * n) 
  (h2 : b + c = a * n) 
  (h3 : a + c = b * n) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  (a + b) * (b + c) * (a + c) / (a * b * c) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2741_274190


namespace NUMINAMATH_CALUDE_smallest_product_of_given_numbers_l2741_274111

theorem smallest_product_of_given_numbers : 
  let numbers : List ℕ := [10, 11, 12, 13, 14]
  let smallest := numbers.minimum?
  let next_smallest := numbers.filter (· ≠ smallest.getD 0) |>.minimum?
  smallest.isSome ∧ next_smallest.isSome → 
  smallest.getD 0 * next_smallest.getD 0 = 110 := by
sorry

end NUMINAMATH_CALUDE_smallest_product_of_given_numbers_l2741_274111


namespace NUMINAMATH_CALUDE_motorcycle_friction_speed_relation_l2741_274106

/-- Proves that the minimum friction coefficient for a motorcycle riding on vertical walls
    is inversely proportional to the square of its speed. -/
theorem motorcycle_friction_speed_relation 
  (m : ℝ) -- mass of the motorcycle
  (g : ℝ) -- acceleration due to gravity
  (r : ℝ) -- radius of the circular room
  (s : ℝ) -- speed of the motorcycle
  (μ : ℝ → ℝ) -- friction coefficient as a function of speed
  (h_positive : m > 0 ∧ g > 0 ∧ r > 0 ∧ s > 0) -- positivity conditions
  (h_equilibrium : ∀ s, μ s * (m * s^2 / r) = m * g) -- equilibrium condition
  : ∃ (k : ℝ), ∀ s, μ s = k / s^2 :=
sorry

end NUMINAMATH_CALUDE_motorcycle_friction_speed_relation_l2741_274106


namespace NUMINAMATH_CALUDE_hand_74_falls_off_after_20_minutes_l2741_274151

/-- Represents a clock hand with its rotation speed and fall-off time. -/
structure ClockHand where
  speed : ℕ
  fallOffTime : ℚ

/-- Represents a clock with multiple hands. -/
def Clock := List ClockHand

/-- Creates a clock with the specified number of hands. -/
def createClock (n : ℕ) : Clock :=
  List.range n |>.map (fun i => { speed := i + 1, fallOffTime := 0 })

/-- Calculates the fall-off time for a specific hand in the clock. -/
def calculateFallOffTime (clock : Clock) (handSpeed : ℕ) : ℚ :=
  sorry

/-- Theorem: The 74th hand in a 150-hand clock falls off after 20 minutes. -/
theorem hand_74_falls_off_after_20_minutes :
  let clock := createClock 150
  calculateFallOffTime clock 74 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_hand_74_falls_off_after_20_minutes_l2741_274151


namespace NUMINAMATH_CALUDE_choir_size_proof_l2741_274166

theorem choir_size_proof : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 8 = 0 ∧ 
  n % 9 = 0 ∧ 
  n % 10 = 0 ∧ 
  n % 11 = 0 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 8 = 0 ∧ m % 9 = 0 ∧ m % 10 = 0 ∧ m % 11 = 0 → m ≥ n) ∧
  n = 1080 :=
by sorry

end NUMINAMATH_CALUDE_choir_size_proof_l2741_274166


namespace NUMINAMATH_CALUDE_chris_candy_distribution_chris_total_candy_l2741_274107

theorem chris_candy_distribution (first_group : Nat) (first_amount : Nat) 
  (second_group : Nat) (remaining_amount : Nat) : Nat :=
  let total_first := first_group * first_amount
  let total_second := second_group * (2 * first_amount)
  total_first + total_second + remaining_amount

theorem chris_total_candy : chris_candy_distribution 10 12 7 50 = 338 := by
  sorry

end NUMINAMATH_CALUDE_chris_candy_distribution_chris_total_candy_l2741_274107


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l2741_274192

theorem stratified_sampling_proportion (total : ℕ) (first_year : ℕ) (second_year : ℕ) 
  (sample_first : ℕ) (sample_second : ℕ) :
  total = first_year + second_year →
  first_year * sample_second = second_year * sample_first →
  sample_first = 6 →
  first_year = 30 →
  second_year = 40 →
  sample_second = 8 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l2741_274192


namespace NUMINAMATH_CALUDE_largest_prime_sum_under_30_l2741_274117

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

theorem largest_prime_sum_under_30 :
  is_prime 19 ∧
  19 < 30 ∧
  is_sum_of_two_primes 19 ∧
  ∀ n : ℕ, is_prime n → n < 30 → is_sum_of_two_primes n → n ≤ 19 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_sum_under_30_l2741_274117


namespace NUMINAMATH_CALUDE_max_book_combination_l2741_274182

theorem max_book_combination (total : ℕ) (math_books logic_books : ℕ → ℕ) : 
  total = 20 →
  (∀ k, math_books k + logic_books k = total) →
  (∀ k, 0 ≤ k ∧ k ≤ 10 → math_books k = 10 - k ∧ logic_books k = 10 + k) →
  (∀ k, 0 ≤ k ∧ k ≤ 10 → Nat.choose (math_books k) 5 * Nat.choose (logic_books k) 5 ≤ (Nat.choose 10 5)^2) :=
by sorry

end NUMINAMATH_CALUDE_max_book_combination_l2741_274182


namespace NUMINAMATH_CALUDE_factorial_products_perfect_square_l2741_274169

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem factorial_products_perfect_square : 
  is_perfect_square (factorial 99 * factorial 100) ∧
  ¬is_perfect_square (factorial 97 * factorial 98) ∧
  ¬is_perfect_square (factorial 97 * factorial 99) ∧
  ¬is_perfect_square (factorial 98 * factorial 99) ∧
  ¬is_perfect_square (factorial 98 * factorial 100) :=
by sorry

end NUMINAMATH_CALUDE_factorial_products_perfect_square_l2741_274169


namespace NUMINAMATH_CALUDE_wire_weight_proportional_l2741_274104

/-- Given that a 25 m roll of wire weighs 5 kg, prove that a 75 m roll of wire weighs 15 kg. -/
theorem wire_weight_proportional (length_short : ℝ) (weight_short : ℝ) (length_long : ℝ) :
  length_short = 25 →
  weight_short = 5 →
  length_long = 75 →
  (length_long / length_short) * weight_short = 15 := by
  sorry

end NUMINAMATH_CALUDE_wire_weight_proportional_l2741_274104


namespace NUMINAMATH_CALUDE_units_produced_today_l2741_274115

/-- Calculates the number of units produced today given previous production data -/
theorem units_produced_today (n : ℕ) (prev_avg : ℝ) (new_avg : ℝ) 
  (h1 : n = 4)
  (h2 : prev_avg = 50)
  (h3 : new_avg = 58) : 
  (n + 1 : ℝ) * new_avg - n * prev_avg = 90 := by
  sorry

end NUMINAMATH_CALUDE_units_produced_today_l2741_274115


namespace NUMINAMATH_CALUDE_right_triangle_proof_l2741_274164

-- Define a structure for a triangle with three angles
structure Triangle where
  α : Real
  β : Real
  γ : Real

-- Define the theorem
theorem right_triangle_proof (t : Triangle) (h : t.γ = t.α + t.β) : 
  t.α = 90 ∨ t.β = 90 ∨ t.γ = 90 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_proof_l2741_274164


namespace NUMINAMATH_CALUDE_fast_food_theorem_l2741_274155

/-- A fast food composition -/
structure FastFood where
  total_mass : ℝ
  fat_percentage : ℝ
  protein_mass : ℝ → ℝ
  mineral_mass : ℝ → ℝ
  carb_mass : ℝ → ℝ

/-- Conditions for the fast food -/
def fast_food_conditions (ff : FastFood) : Prop :=
  ff.total_mass = 500 ∧
  ff.fat_percentage = 0.05 ∧
  (∀ x, ff.protein_mass x = 4 * ff.mineral_mass x) ∧
  (∀ x, (ff.protein_mass x + ff.carb_mass x) / ff.total_mass ≤ 0.85)

/-- Theorem about the mass of fat and maximum carbohydrates in the fast food -/
theorem fast_food_theorem (ff : FastFood) (h : fast_food_conditions ff) :
  ff.fat_percentage * ff.total_mass = 25 ∧
  ∃ x, ff.carb_mass x = 225 ∧ 
    ∀ y, ff.carb_mass y ≤ ff.carb_mass x :=
by sorry

end NUMINAMATH_CALUDE_fast_food_theorem_l2741_274155


namespace NUMINAMATH_CALUDE_lcm_hcf_relation_l2741_274174

theorem lcm_hcf_relation (x y : ℕ+) (h_lcm : Nat.lcm x y = 1637970) (h_hcf : Nat.gcd x y = 210) (h_x : x = 10780) : y = 31910 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_relation_l2741_274174


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2741_274170

theorem sqrt_equation_solution (y : ℝ) :
  (y > 2) →  -- This condition is necessary to ensure the square root is defined
  (Real.sqrt (8 * y) / Real.sqrt (4 * (y - 2)) = 3) →
  y = 18 / 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2741_274170


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l2741_274197

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_derivative_at_one : 
  deriv f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l2741_274197


namespace NUMINAMATH_CALUDE_fifteen_initial_points_theorem_l2741_274188

/-- The number of points after n iterations of the marking process -/
def points_after_iteration (initial_points : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial_points
  | k + 1 => 2 * (points_after_iteration initial_points k) - 1

/-- The theorem stating that 15 initial points result in 225 points after 4 iterations -/
theorem fifteen_initial_points_theorem :
  ∃ (initial_points : ℕ), 
    initial_points > 0 ∧ 
    points_after_iteration initial_points 4 = 225 ∧ 
    initial_points = 15 := by
  sorry


end NUMINAMATH_CALUDE_fifteen_initial_points_theorem_l2741_274188


namespace NUMINAMATH_CALUDE_tangency_points_on_sphere_l2741_274112

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Represents a point in 3D space -/
def Point := ℝ × ℝ × ℝ

/-- Predicate to check if two spheres are tangent -/
def are_tangent (s1 s2 : Sphere) : Prop := sorry

/-- Function to get the tangency point of two spheres -/
def tangency_point (s1 s2 : Sphere) : Point := sorry

/-- Predicate to check if a point lies on a sphere -/
def point_on_sphere (p : Point) (s : Sphere) : Prop := sorry

theorem tangency_points_on_sphere 
  (s1 s2 s3 s4 : Sphere) 
  (h1 : are_tangent s1 s2) (h2 : are_tangent s1 s3) (h3 : are_tangent s1 s4)
  (h4 : are_tangent s2 s3) (h5 : are_tangent s2 s4) (h6 : are_tangent s3 s4) :
  ∃ (s : Sphere), 
    point_on_sphere (tangency_point s1 s2) s ∧
    point_on_sphere (tangency_point s1 s3) s ∧
    point_on_sphere (tangency_point s1 s4) s ∧
    point_on_sphere (tangency_point s2 s3) s ∧
    point_on_sphere (tangency_point s2 s4) s ∧
    point_on_sphere (tangency_point s3 s4) s :=
  sorry

end NUMINAMATH_CALUDE_tangency_points_on_sphere_l2741_274112


namespace NUMINAMATH_CALUDE_part1_part2_part3_l2741_274153

-- Define the variables and constants
variable (x y : ℝ)  -- x: quantity of vegetable A, y: quantity of vegetable B
def total_weight : ℝ := 40
def total_cost : ℝ := 180
def wholesale_price_A : ℝ := 4.8
def wholesale_price_B : ℝ := 4
def retail_price_A : ℝ := 7.2
def retail_price_B : ℝ := 5.6
def new_total_weight : ℝ := 80
def min_profit : ℝ := 176

-- Part 1
theorem part1 : 
  x + y = total_weight ∧ 
  wholesale_price_A * x + wholesale_price_B * y = total_cost → 
  x = 25 ∧ y = 15 := by sorry

-- Part 2
def m (n : ℝ) : ℝ := wholesale_price_A * n + wholesale_price_B * (new_total_weight - n)

theorem part2 : m n = 0.8 * n + 320 := by sorry

-- Part 3
def profit (n : ℝ) : ℝ := (retail_price_A - wholesale_price_A) * n + 
                           (retail_price_B - wholesale_price_B) * (new_total_weight - n)

theorem part3 : 
  ∀ n : ℝ, profit n ≥ min_profit → n ≥ 60 := by sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l2741_274153


namespace NUMINAMATH_CALUDE_correct_calculation_l2741_274179

theorem correct_calculation (x y : ℝ) : 2 * x * y^2 - x * y^2 = x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2741_274179


namespace NUMINAMATH_CALUDE_volume_ratio_cubes_l2741_274135

/-- The ratio of the volume of a cube with edge length 4 inches to the volume of a cube with edge length 2 feet -/
theorem volume_ratio_cubes : 
  let small_edge_inches : ℚ := 4
  let large_edge_feet : ℚ := 2
  let inches_per_foot : ℚ := 12
  let small_edge_feet : ℚ := small_edge_inches / inches_per_foot
  let volume_ratio : ℚ := (small_edge_feet / large_edge_feet) ^ 3
  volume_ratio = 1 / 216 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_cubes_l2741_274135


namespace NUMINAMATH_CALUDE_distance_calculation_l2741_274103

/-- The distance between Xiao Ming's home and his grandmother's house -/
def distance_to_grandma : ℝ := 36

/-- Xiao Ming's speed in km/h -/
def xiao_ming_speed : ℝ := 12

/-- Father's speed in km/h -/
def father_speed : ℝ := 36

/-- Time Xiao Ming departs before his father in hours -/
def time_before_father : ℝ := 2.5

/-- Time father arrives after Xiao Ming in hours -/
def time_after_xiao_ming : ℝ := 0.5

theorem distance_calculation :
  ∃ (t : ℝ),
    t > 0 ∧
    distance_to_grandma = father_speed * t ∧
    distance_to_grandma = xiao_ming_speed * (t + time_before_father - time_after_xiao_ming) :=
by
  sorry

#check distance_calculation

end NUMINAMATH_CALUDE_distance_calculation_l2741_274103


namespace NUMINAMATH_CALUDE_point_not_on_graph_l2741_274162

def inverse_proportion (x y : ℝ) : Prop := x * y = 6

theorem point_not_on_graph :
  ¬(inverse_proportion 1 5) ∧ 
  (inverse_proportion (-2) (-3)) ∧ 
  (inverse_proportion (-3) (-2)) ∧ 
  (inverse_proportion 4 1.5) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_graph_l2741_274162


namespace NUMINAMATH_CALUDE_program_arrangements_l2741_274121

/-- The number of ways to arrange n items in k positions --/
def arrangement (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to insert 3 new programs into a list of 10 existing programs --/
theorem program_arrangements : 
  arrangement 11 3 + arrangement 3 2 * arrangement 11 2 + arrangement 3 3 * arrangement 11 1 = 1716 := by
  sorry

end NUMINAMATH_CALUDE_program_arrangements_l2741_274121


namespace NUMINAMATH_CALUDE_smith_bought_six_boxes_l2741_274193

/-- Calculates the number of new boxes of markers bought by Mr. Smith -/
def new_boxes_bought (initial_markers : ℕ) (markers_per_box : ℕ) (final_markers : ℕ) : ℕ :=
  (final_markers - initial_markers) / markers_per_box

/-- Proves that Mr. Smith bought 6 new boxes of markers -/
theorem smith_bought_six_boxes :
  new_boxes_bought 32 9 86 = 6 := by
  sorry

#eval new_boxes_bought 32 9 86

end NUMINAMATH_CALUDE_smith_bought_six_boxes_l2741_274193


namespace NUMINAMATH_CALUDE_sequence_properties_l2741_274184

def sequence_a (n : ℕ) : ℝ := 6 * 2^(n-1) - 3

def sum_S (n : ℕ) : ℝ := 6 * 2^n - 3 * n - 6

theorem sequence_properties :
  let a := sequence_a
  let S := sum_S
  (∀ n : ℕ, n ≥ 1 → S n = 2 * a n - 3 * n) →
  (a 1 = 3 ∧
   ∀ n : ℕ, a (n + 1) = 2 * a n + 3 ∧
   ∀ n : ℕ, n ≥ 1 → a n = 6 * 2^(n-1) - 3 ∧
   ∀ n : ℕ, n ≥ 1 → S n = 6 * 2^n - 3 * n - 6) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2741_274184


namespace NUMINAMATH_CALUDE_area_ratio_incenter_centroids_l2741_274165

/-- Triangle type -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Area of a triangle -/
def area (t : Triangle) : ℝ := sorry

theorem area_ratio_incenter_centroids 
  (ABC : Triangle) 
  (P : ℝ × ℝ) 
  (G₁ G₂ G₃ : ℝ × ℝ) :
  P = incenter ABC →
  G₁ = centroid (Triangle.mk P ABC.B ABC.C) →
  G₂ = centroid (Triangle.mk ABC.A P ABC.C) →
  G₃ = centroid (Triangle.mk ABC.A ABC.B P) →
  area (Triangle.mk G₁ G₂ G₃) = (1 / 9) * area ABC :=
by sorry

end NUMINAMATH_CALUDE_area_ratio_incenter_centroids_l2741_274165


namespace NUMINAMATH_CALUDE_parallelogram_ABCD_area_l2741_274128

-- Define the parallelogram vertices
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (5, 1)
def C : ℝ × ℝ := (7, 4)
def D : ℝ × ℝ := (3, 4)

-- Define a function to calculate the area of a parallelogram given two vectors
def parallelogramArea (v1 v2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  abs (x1 * y2 - x2 * y1)

-- Theorem statement
theorem parallelogram_ABCD_area :
  parallelogramArea (B.1 - A.1, B.2 - A.2) (D.1 - A.1, D.2 - A.2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_ABCD_area_l2741_274128


namespace NUMINAMATH_CALUDE_quadratic_sum_equals_113_l2741_274177

theorem quadratic_sum_equals_113 (x y : ℝ) 
  (eq1 : 3*x + 2*y = 7) 
  (eq2 : 2*x + 3*y = 8) : 
  13*x^2 + 22*x*y + 13*y^2 = 113 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_equals_113_l2741_274177


namespace NUMINAMATH_CALUDE_problem_solution_l2741_274114

-- Define the set B
def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1) 2, x^2 - 2*x - m ≤ 0}

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | (x - 2*a) * (x - (a + 1)) ≤ 0}

-- State the theorem
theorem problem_solution :
  (B = Set.Ici 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ B → x ∈ A a) ∧ (∃ x : ℝ, x ∈ A a ∧ x ∉ B) → a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2741_274114


namespace NUMINAMATH_CALUDE_peach_baskets_l2741_274152

theorem peach_baskets (peaches_per_basket : ℕ) (eaten_peaches : ℕ) 
  (small_boxes : ℕ) (peaches_per_small_box : ℕ) :
  peaches_per_basket = 25 →
  eaten_peaches = 5 →
  small_boxes = 8 →
  peaches_per_small_box = 15 →
  (small_boxes * peaches_per_small_box + eaten_peaches) / peaches_per_basket = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_peach_baskets_l2741_274152


namespace NUMINAMATH_CALUDE_sum_of_first_100_terms_l2741_274176

-- Define the function f
def f (n : ℕ) : ℤ :=
  if n % 2 = 1 then n^2 else -(n^2)

-- Define the sequence a_n
def a (n : ℕ) : ℤ := f n + f (n + 1)

-- State the theorem
theorem sum_of_first_100_terms :
  (Finset.range 100).sum (λ i => a (i + 1)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_100_terms_l2741_274176


namespace NUMINAMATH_CALUDE_simplify_expression_l2741_274101

theorem simplify_expression (a : ℝ) :
  (((a ^ 16) ^ (1 / 8)) ^ (1 / 4)) ^ 3 * (((a ^ 16) ^ (1 / 4)) ^ (1 / 8)) ^ 3 = a ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2741_274101


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2741_274129

theorem simplify_square_roots : 
  (Real.sqrt 450 / Real.sqrt 400) + (Real.sqrt 98 / Real.sqrt 56) = (3 + 2 * Real.sqrt 7) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2741_274129


namespace NUMINAMATH_CALUDE_dresser_contents_l2741_274158

/-- Given a dresser with pants, shorts, and shirts in the ratio 7 : 7 : 10,
    prove that if there are 14 pants, there are 20 shirts. -/
theorem dresser_contents (pants shorts shirts : ℕ) : 
  pants = 14 →
  pants * 10 = shirts * 7 →
  shirts = 20 := by
  sorry

end NUMINAMATH_CALUDE_dresser_contents_l2741_274158


namespace NUMINAMATH_CALUDE_complex_magnitude_l2741_274138

theorem complex_magnitude (i : ℂ) (z : ℂ) : 
  i * i = -1 → z = (1 + i) / 2 → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2741_274138


namespace NUMINAMATH_CALUDE_max_value_on_circle_l2741_274189

theorem max_value_on_circle (x y : ℝ) : 
  (x - 3)^2 + (y - 4)^2 = 9 → 
  ∃ (z : ℝ), z = 3*x + 4*y ∧ z ≤ 40 ∧ ∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 4)^2 = 9 ∧ 3*x₀ + 4*y₀ = 40 :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l2741_274189


namespace NUMINAMATH_CALUDE_gcd_108_45_l2741_274194

theorem gcd_108_45 : Nat.gcd 108 45 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_108_45_l2741_274194


namespace NUMINAMATH_CALUDE_system_solution_proof_l2741_274173

theorem system_solution_proof : ∃ (x y : ℝ), 
  (2 * x + 7 * y = -6) ∧ 
  (2 * x - 5 * y = 18) ∧ 
  (x = 4) ∧ 
  (y = -2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_proof_l2741_274173


namespace NUMINAMATH_CALUDE_girls_in_class_l2741_274185

/-- Given a class with a 3:4 ratio of girls to boys and 35 total students,
    prove that the number of girls is 15. -/
theorem girls_in_class (g b : ℕ) : 
  g + b = 35 →  -- Total number of students
  4 * g = 3 * b →  -- Ratio of girls to boys is 3:4
  g = 15 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l2741_274185


namespace NUMINAMATH_CALUDE_winnie_the_pooh_fall_damage_ratio_l2741_274171

/-- The ratio of damages in Winnie-the-Pooh's fall -/
theorem winnie_the_pooh_fall_damage_ratio 
  (g M τ H : ℝ) 
  (n k : ℝ) 
  (h : ℝ := H / n) 
  (V_I : ℝ := Real.sqrt (2 * g * H)) 
  (V_1 : ℝ := Real.sqrt (2 * g * h)) 
  (V_1_prime : ℝ := (1 / k) * Real.sqrt (2 * g * h)) 
  (V_II : ℝ := Real.sqrt ((1 / k^2) * 2 * g * h + 2 * g * (H - h))) 
  (I_I : ℝ := M * V_I * τ) 
  (I_II : ℝ := M * τ * ((V_1 - V_1_prime) + V_II)) 
  (hg : g > 0) 
  (hM : M > 0) 
  (hτ : τ > 0) 
  (hH : H > 0) 
  (hn : n > 0) 
  (hk : k > 0) : 
  I_II / I_I = 5 / 4 := by
  sorry


end NUMINAMATH_CALUDE_winnie_the_pooh_fall_damage_ratio_l2741_274171


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l2741_274178

theorem infinitely_many_solutions :
  ∃ f : ℕ → ℕ × ℕ, ∀ n : ℕ,
    let (a, b) := f n
    (a > 0 ∧ b > 0) ∧ a^2 - b^2 = a * b - 1 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l2741_274178


namespace NUMINAMATH_CALUDE_sin_alpha_equals_half_l2741_274136

/-- If the terminal side of angle α passes through the point (-√3, 1), then sin α = 1/2 -/
theorem sin_alpha_equals_half (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -Real.sqrt 3 ∧ t * Real.sin α = 1) → 
  Real.sin α = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_equals_half_l2741_274136


namespace NUMINAMATH_CALUDE_first_caterer_cheaper_at_17_l2741_274154

/-- The least number of people for which the first caterer is cheaper -/
def least_people_first_caterer_cheaper : ℕ := 17

/-- Cost function for the first caterer -/
def cost_first_caterer (people : ℕ) : ℚ := 200 + 18 * people

/-- Cost function for the second caterer -/
def cost_second_caterer (people : ℕ) : ℚ := 250 + 15 * people

/-- Theorem stating that 17 is the least number of people for which the first caterer is cheaper -/
theorem first_caterer_cheaper_at_17 :
  (∀ n : ℕ, n < least_people_first_caterer_cheaper →
    cost_first_caterer n ≥ cost_second_caterer n) ∧
  cost_first_caterer least_people_first_caterer_cheaper < cost_second_caterer least_people_first_caterer_cheaper :=
by sorry

end NUMINAMATH_CALUDE_first_caterer_cheaper_at_17_l2741_274154


namespace NUMINAMATH_CALUDE_product_repeating_third_twelve_l2741_274127

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1/3

/-- The product of 0.333... and 12 is 4 --/
theorem product_repeating_third_twelve : repeating_third * 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_repeating_third_twelve_l2741_274127


namespace NUMINAMATH_CALUDE_compare_exponentials_l2741_274125

theorem compare_exponentials (h1 : 0 < 0.7) (h2 : 0.7 < 0.8) (h3 : 0.8 < 1) :
  0.8^0.7 > 0.7^0.7 ∧ 0.7^0.7 > 0.7^0.8 := by
  sorry

end NUMINAMATH_CALUDE_compare_exponentials_l2741_274125


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_8_l2741_274140

theorem binomial_coefficient_20_8 :
  let n : ℕ := 20
  let k : ℕ := 8
  let binomial := Nat.choose
  binomial 18 5 = 8568 →
  binomial 18 7 = 31824 →
  binomial n k = 83656 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_8_l2741_274140
