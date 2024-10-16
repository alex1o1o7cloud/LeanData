import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l938_93886

theorem equation_solution (y : ℝ) (h : y ≠ 2) : 
  (y^2 - 10*y + 24) / (y - 2) + (4*y^2 + 8*y - 48) / (4*y - 8) = 0 ↔ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l938_93886


namespace NUMINAMATH_CALUDE_income_comparison_l938_93882

/-- Given that Mart's income is 60% more than Tim's income, and Tim's income is 50% less than Juan's income, 
    prove that Mart's income is 80% of Juan's income. -/
theorem income_comparison (tim juan mart : ℝ) 
  (h1 : mart = tim + 0.6 * tim) 
  (h2 : tim = juan - 0.5 * juan) : 
  mart = 0.8 * juan := by sorry

end NUMINAMATH_CALUDE_income_comparison_l938_93882


namespace NUMINAMATH_CALUDE_will_toy_purchase_l938_93876

def max_toys_purchasable (initial_amount : ℕ) (game_cost : ℕ) (toy_cost : ℕ) : ℕ :=
  ((initial_amount - game_cost) / toy_cost)

theorem will_toy_purchase : max_toys_purchasable 57 27 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_will_toy_purchase_l938_93876


namespace NUMINAMATH_CALUDE_golden_ratio_in_line_segment_l938_93838

/-- Given a line segment AC divided by point B, prove that if BC/AB = AB/AC = k, then k = (√5 - 1) / 2 -/
theorem golden_ratio_in_line_segment (A B C : ℝ) (k : ℝ) 
  (h1 : A < B) (h2 : B < C)
  (h3 : (C - B) / (B - A) = k)
  (h4 : (B - A) / (C - A) = k) :
  k = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_in_line_segment_l938_93838


namespace NUMINAMATH_CALUDE_number_for_B_l938_93872

/-- Given that the number for A is a, and the number for B is 1 less than twice the number for A,
    prove that the number for B can be expressed as 2a - 1. -/
theorem number_for_B (a : ℝ) : 2 * a - 1 = 2 * a - 1 := by sorry

end NUMINAMATH_CALUDE_number_for_B_l938_93872


namespace NUMINAMATH_CALUDE_chord_length_problem_l938_93845

/-- The chord length cut by a line on a circle -/
def chord_length (circle_center : ℝ × ℝ) (circle_radius : ℝ) (line : ℝ → ℝ → ℝ) : ℝ :=
  2 * circle_radius

/-- The problem statement -/
theorem chord_length_problem :
  let circle_center := (3, 0)
  let circle_radius := 3
  let line := fun x y => 3 * x - 4 * y - 9
  chord_length circle_center circle_radius line = 6 := by
  sorry


end NUMINAMATH_CALUDE_chord_length_problem_l938_93845


namespace NUMINAMATH_CALUDE_divisibility_criterion_37_l938_93866

/-- Represents a function that divides a positive integer into three-digit segments from right to left -/
def segmentNumber (n : ℕ+) : List ℕ :=
  sorry

/-- Theorem: A positive integer is divisible by 37 if and only if the sum of its three-digit segments is divisible by 37 -/
theorem divisibility_criterion_37 (n : ℕ+) :
  37 ∣ n ↔ 37 ∣ (segmentNumber n).sum :=
by sorry

end NUMINAMATH_CALUDE_divisibility_criterion_37_l938_93866


namespace NUMINAMATH_CALUDE_plane_trip_distance_l938_93817

/-- Proves that if a person takes a trip a certain number of times and travels a total distance,
    then the distance of each trip is the total distance divided by the number of trips. -/
theorem plane_trip_distance (num_trips : ℝ) (total_distance : ℝ) 
    (h1 : num_trips = 32) 
    (h2 : total_distance = 8192) : 
  total_distance / num_trips = 256 := by
  sorry

#check plane_trip_distance

end NUMINAMATH_CALUDE_plane_trip_distance_l938_93817


namespace NUMINAMATH_CALUDE_prob_two_sixes_is_one_thirty_sixth_l938_93851

/-- A fair six-sided die -/
structure FairDie :=
  (sides : Fin 6)

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll (d : FairDie) (n : Fin 6) : ℚ := 1 / 6

/-- The probability of rolling two consecutive sixes -/
def prob_two_sixes (d : FairDie) : ℚ :=
  (prob_single_roll d 5) * (prob_single_roll d 5)

/-- Theorem: The probability of rolling two consecutive sixes with a fair six-sided die is 1/36 -/
theorem prob_two_sixes_is_one_thirty_sixth (d : FairDie) :
  prob_two_sixes d = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_sixes_is_one_thirty_sixth_l938_93851


namespace NUMINAMATH_CALUDE_sum_of_squares_l938_93843

theorem sum_of_squares (a b c : ℝ) : 
  a + b + c = 21 → 
  a * b + b * c + a * c = 100 → 
  a^2 + b^2 + c^2 = 241 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l938_93843


namespace NUMINAMATH_CALUDE_work_completion_time_l938_93850

/-- Given that person A can complete a work in 30 days, and persons A and B together complete 1/9 of the work in 2 days, prove that person B can complete the work alone in 45 days. -/
theorem work_completion_time (a b : ℝ) (h1 : a = 30) 
  (h2 : 2 * (1 / a + 1 / b) = 1 / 9) : b = 45 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l938_93850


namespace NUMINAMATH_CALUDE_min_value_theorem_l938_93821

theorem min_value_theorem (x A B : ℝ) (hx : x > 0) (hA : A > 0) (hB : B > 0)
  (eq1 : x^2 + 1/x^2 = A) (eq2 : x - 1/x = B) :
  ∀ y, y > 0 → y^2 + 1/y^2 = A → y - 1/y = B → (A + 1) / B ≥ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l938_93821


namespace NUMINAMATH_CALUDE_binomial_10_0_l938_93806

theorem binomial_10_0 : (10 : ℕ).choose 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_0_l938_93806


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l938_93869

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I * (a - Complex.I * (a + 2)) = 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l938_93869


namespace NUMINAMATH_CALUDE_brick_length_proof_l938_93804

/-- Given a courtyard and bricks with specific dimensions, prove the length of each brick -/
theorem brick_length_proof (courtyard_length : ℝ) (courtyard_breadth : ℝ) 
  (brick_breadth : ℝ) (total_bricks : ℕ) :
  courtyard_length = 20 →
  courtyard_breadth = 16 →
  brick_breadth = 0.1 →
  total_bricks = 16000 →
  (courtyard_length * courtyard_breadth * 10000) / (total_bricks * brick_breadth * 100) = 20 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_proof_l938_93804


namespace NUMINAMATH_CALUDE_company_profit_and_assignment_l938_93802

/-- Represents the profit calculation for a company with two products. -/
def CompanyProfit (totalWorkers : ℕ) (profitA profitB : ℚ) (decreaseRate : ℚ) : Prop :=
  ∃ x : ℕ,
    x ≤ totalWorkers ∧
    let workersA := totalWorkers - x
    let outputA := 2 * workersA
    let outputB := x
    let profitPerUnitB := profitB - decreaseRate * x
    let totalProfitA := profitA * outputA
    let totalProfitB := profitPerUnitB * outputB
    totalProfitA = totalProfitB + 650 ∧
    totalProfitA + totalProfitB = 2650

/-- Represents the optimal worker assignment when introducing a third product. -/
def OptimalAssignment (totalWorkers : ℕ) (profitA profitB profitC : ℚ) (decreaseRate : ℚ) : Prop :=
  ∃ m : ℕ,
    m ≤ totalWorkers ∧
    let workersA := m
    let workersC := 2 * m
    let workersB := totalWorkers - workersA - workersC
    workersA + workersB + workersC = totalWorkers ∧
    let outputA := 2 * workersA
    let outputB := workersB
    let outputC := workersC
    outputA = outputC ∧
    let profitPerUnitB := profitB - decreaseRate * workersB
    let totalProfit := profitA * outputA + profitPerUnitB * outputB + profitC * outputC
    totalProfit = 2650 ∧
    m = 10

/-- Theorem stating the company's profit and optimal assignment. -/
theorem company_profit_and_assignment :
  CompanyProfit 65 15 120 2 ∧
  OptimalAssignment 65 15 120 30 2 :=
sorry

end NUMINAMATH_CALUDE_company_profit_and_assignment_l938_93802


namespace NUMINAMATH_CALUDE_count_divisible_numbers_eq_179_l938_93826

/-- The count of five-digit numbers exactly divisible by 6, 7, 8, and 9 -/
def count_divisible_numbers : ℕ :=
  let lcm := Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))
  let lower_bound := ((10000 + lcm - 1) / lcm : ℕ)
  let upper_bound := (99999 / lcm : ℕ)
  upper_bound - lower_bound + 1

theorem count_divisible_numbers_eq_179 : count_divisible_numbers = 179 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_eq_179_l938_93826


namespace NUMINAMATH_CALUDE_class_payment_problem_l938_93832

theorem class_payment_problem (total_students : ℕ) (full_payment half_payment total_collected : ℚ) 
  (h1 : total_students = 25)
  (h2 : full_payment = 50)
  (h3 : half_payment = 25)
  (h4 : total_collected = 1150)
  (h5 : ∃ (full_payers half_payers : ℕ), 
    full_payers + half_payers = total_students ∧ 
    full_payers * full_payment + half_payers * half_payment = total_collected) :
  ∃ (half_payers : ℕ), half_payers = 4 := by
sorry

end NUMINAMATH_CALUDE_class_payment_problem_l938_93832


namespace NUMINAMATH_CALUDE_beri_always_wins_l938_93837

def has_integer_solutions (a b : ℕ) : Prop :=
  ∃ x : ℤ, x^2 - a*x + b = 0 ∨ x^2 - b*x + a = 0

theorem beri_always_wins :
  ∀ a ∈ Finset.range 2021,
    ∃ b ∈ Finset.range 2021,
      b ≠ a ∧ has_integer_solutions a b :=
by sorry

end NUMINAMATH_CALUDE_beri_always_wins_l938_93837


namespace NUMINAMATH_CALUDE_unique_solution_for_k_squared_minus_2016_equals_3_to_n_l938_93879

theorem unique_solution_for_k_squared_minus_2016_equals_3_to_n :
  ∃! (k n : ℕ), k > 0 ∧ n > 0 ∧ k^2 - 2016 = 3^n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_k_squared_minus_2016_equals_3_to_n_l938_93879


namespace NUMINAMATH_CALUDE_range_of_b_l938_93897

theorem range_of_b (a b c : ℝ) (h1 : a * c = b^2) (h2 : a + b + c = 3) :
  -3 ≤ b ∧ b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_l938_93897


namespace NUMINAMATH_CALUDE_line_through_A_equal_intercepts_line_BC_equation_l938_93820

-- Define the point A
def A : ℝ × ℝ := (2, 1)

-- Part 1: Line through A with equal intercepts
theorem line_through_A_equal_intercepts :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (a * A.1 + b * A.2 + c = 0) ∧
  (a + b + c = 0) ∧
  (a = 1 ∧ b = 1 ∧ c = -3) := by sorry

-- Part 2: Triangle ABC
theorem line_BC_equation (B C : ℝ × ℝ) :
  -- Given conditions
  (B.1 - B.2 = 0) →  -- B is on the line x - y = 0
  (2 * ((A.1 + B.1) / 2) + ((A.2 + B.2) / 2) - 1 = 0) →  -- CM is on 2x + y - 1 = 0
  (C.1 + C.2 - 3 = 0) →  -- C is on x + y - 3 = 0
  (2 * C.1 + C.2 - 1 = 0) →  -- C is on 2x + y - 1 = 0
  -- Conclusion
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (a * B.1 + b * B.2 + c = 0) ∧
  (a * C.1 + b * C.2 + c = 0) ∧
  (a = 6 ∧ b = 1 ∧ c = 7) := by sorry

end NUMINAMATH_CALUDE_line_through_A_equal_intercepts_line_BC_equation_l938_93820


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l938_93854

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a line passes through a point
def passes_through (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

-- Define a function to check if a line has equal intercepts
def has_equal_intercepts (l : Line) : Prop :=
  l.a = l.b ∨ (l.a = -l.b ∧ l.c = 0)

-- State the theorem
theorem line_through_point_with_equal_intercepts :
  ∀ l : Line,
    passes_through l 3 (-6) →
    has_equal_intercepts l →
    (l = Line.mk 1 1 3 ∨ l = Line.mk 2 1 0) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l938_93854


namespace NUMINAMATH_CALUDE_nine_bounces_before_pocket_l938_93800

/-- Represents a rectangular pool table -/
structure PoolTable where
  width : ℝ
  height : ℝ

/-- Represents a ball's position and direction -/
structure Ball where
  x : ℝ
  y : ℝ
  dx : ℝ
  dy : ℝ

/-- Counts the number of wall bounces before the ball enters a corner pocket -/
def countBounces (table : PoolTable) (ball : Ball) : ℕ :=
  sorry

/-- Theorem stating that a ball on a 12x10 table bounces 9 times before entering a pocket -/
theorem nine_bounces_before_pocket (table : PoolTable) (ball : Ball) :
  table.width = 12 ∧ table.height = 10 ∧ 
  ball.x = 0 ∧ ball.y = 0 ∧ ball.dx = 1 ∧ ball.dy = 1 →
  countBounces table ball = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_bounces_before_pocket_l938_93800


namespace NUMINAMATH_CALUDE_bombardment_death_percentage_l938_93839

/-- The percentage of people who died by bombardment in a Sri Lankan village --/
def bombardment_percentage (initial_population final_population : ℕ) (departure_rate : ℚ) : ℚ :=
  let x := (initial_population - final_population / (1 - departure_rate)) / initial_population
  x * 100

/-- Theorem stating the percentage of people who died by bombardment --/
theorem bombardment_death_percentage :
  let initial_population : ℕ := 4399
  let final_population : ℕ := 3168
  let departure_rate : ℚ := 1/5
  abs (bombardment_percentage initial_population final_population departure_rate - 9.98) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_bombardment_death_percentage_l938_93839


namespace NUMINAMATH_CALUDE_twelve_lines_formed_l938_93892

/-- A configuration of points in a plane -/
structure PointConfiguration where
  total_points : ℕ
  collinear_points : ℕ
  noncollinear_points : ℕ
  h_total : total_points = collinear_points + noncollinear_points
  h_collinear : collinear_points ≥ 3
  h_noncollinear : noncollinear_points ≥ 0

/-- The number of lines formed by a given point configuration -/
def num_lines (config : PointConfiguration) : ℕ :=
  1 + config.collinear_points * config.noncollinear_points + 
  (config.noncollinear_points * (config.noncollinear_points - 1)) / 2

/-- Theorem: In the given configuration, 12 lines can be formed -/
theorem twelve_lines_formed (config : PointConfiguration) 
  (h1 : config.total_points = 7)
  (h2 : config.collinear_points = 5)
  (h3 : config.noncollinear_points = 2) :
  num_lines config = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_lines_formed_l938_93892


namespace NUMINAMATH_CALUDE_total_seeds_planted_l938_93811

theorem total_seeds_planted (num_flowerbeds : ℕ) (seeds_per_flowerbed : ℕ) 
  (h1 : num_flowerbeds = 8) 
  (h2 : seeds_per_flowerbed = 4) : 
  num_flowerbeds * seeds_per_flowerbed = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_planted_l938_93811


namespace NUMINAMATH_CALUDE_raspberry_pie_degrees_l938_93891

/-- Given a class of students with pie preferences, calculate the degrees for raspberry pie in a pie chart. -/
theorem raspberry_pie_degrees (total_students : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ) 
  (h1 : total_students = 48)
  (h2 : chocolate = 18)
  (h3 : apple = 10)
  (h4 : blueberry = 8)
  (h5 : (total_students - (chocolate + apple + blueberry)) % 2 = 0) :
  let raspberry := (total_students - (chocolate + apple + blueberry)) / 2
  (raspberry : ℚ) / total_students * 360 = 45 := by
  sorry

end NUMINAMATH_CALUDE_raspberry_pie_degrees_l938_93891


namespace NUMINAMATH_CALUDE_expression_approximately_equal_to_0_2436_l938_93815

-- Define the expression
def expression : ℚ := (108 * 3 - (108 + 92)) / (92 * 7 - (45 * 3))

-- State the theorem
theorem expression_approximately_equal_to_0_2436 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.00005 ∧ |expression - 0.2436| < ε := by
  sorry

end NUMINAMATH_CALUDE_expression_approximately_equal_to_0_2436_l938_93815


namespace NUMINAMATH_CALUDE_dennis_initial_amount_l938_93807

theorem dennis_initial_amount (shirt_cost change_received : ℕ) : 
  shirt_cost = 27 → change_received = 23 → shirt_cost + change_received = 50 := by
  sorry

end NUMINAMATH_CALUDE_dennis_initial_amount_l938_93807


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_angles_l938_93861

-- Define a cyclic quadrilateral
def CyclicQuadrilateral (a b c d : ℝ) : Prop :=
  a + c = 180 ∧ b + d = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

-- Define an arithmetic progression
def ArithmeticProgression (a b c d : ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ b - a = r ∧ c - b = r ∧ d - c = r

-- Define a geometric progression
def GeometricProgression (a b c d : ℝ) : Prop :=
  ∃ (q : ℝ), q ≠ 1 ∧ b / a = q ∧ c / b = q ∧ d / c = q

theorem cyclic_quadrilateral_angles :
  (∃ (a b c d : ℝ), CyclicQuadrilateral a b c d ∧ ArithmeticProgression a b c d) ∧
  (¬ ∃ (a b c d : ℝ), CyclicQuadrilateral a b c d ∧ GeometricProgression a b c d) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_angles_l938_93861


namespace NUMINAMATH_CALUDE_area_AEHF_is_twelve_l938_93827

/-- Rectangle ABCD with dimensions 5x6 -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

/-- Point on a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Definition of rectangle ABCD -/
def rect_ABCD : Rectangle :=
  { width := 5, height := 6 }

/-- Point A at (0,0) -/
def point_A : Point :=
  { x := 0, y := 0 }

/-- Point E on CD, 3 units from D -/
def point_E : Point :=
  { x := 3, y := rect_ABCD.height }

/-- Point F on AB, 2 units from A -/
def point_F : Point :=
  { x := 2, y := 0 }

/-- Area of rectangle AEHF -/
def area_AEHF : ℝ :=
  (point_E.x - point_A.x) * (point_E.y - point_A.y)

/-- Theorem stating that the area of rectangle AEHF is 12 square units -/
theorem area_AEHF_is_twelve : area_AEHF = 12 := by
  sorry

end NUMINAMATH_CALUDE_area_AEHF_is_twelve_l938_93827


namespace NUMINAMATH_CALUDE_no_triangle_with_heights_1_2_3_l938_93810

theorem no_triangle_with_heights_1_2_3 : 
  ¬ ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive side lengths
    (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧  -- triangle inequality
    (1 : ℝ) = (2 * (a * b * c).sqrt) / (b * c) ∧  -- height 1
    (2 : ℝ) = (2 * (a * b * c).sqrt) / (a * c) ∧  -- height 2
    (3 : ℝ) = (2 * (a * b * c).sqrt) / (a * b) :=  -- height 3
by sorry


end NUMINAMATH_CALUDE_no_triangle_with_heights_1_2_3_l938_93810


namespace NUMINAMATH_CALUDE_determinant_difference_l938_93830

theorem determinant_difference (a b c d : ℝ) :
  Matrix.det !![a, b; c, d] = 15 →
  Matrix.det !![3*a, 3*b; 3*c, 3*d] - Matrix.det !![3*b, 3*a; 3*d, 3*c] = 270 := by
sorry

end NUMINAMATH_CALUDE_determinant_difference_l938_93830


namespace NUMINAMATH_CALUDE_another_divisor_of_44404_l938_93847

theorem another_divisor_of_44404 (n : Nat) (h1 : n = 44404) 
  (h2 : 12 ∣ n) (h3 : 48 ∣ n) (h4 : 74 ∣ n) (h5 : 100 ∣ n) : 
  199 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_another_divisor_of_44404_l938_93847


namespace NUMINAMATH_CALUDE_find_a_solve_inequality_l938_93893

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1|

-- Theorem 1: Prove the value of a
theorem find_a : 
  ∀ a : ℝ, (∀ x : ℝ, f a x ≤ 2 ↔ -1/2 ≤ x ∧ x ≤ 3/2) → a = 2 :=
by sorry

-- Theorem 2: Prove the solution set of the inequality
theorem solve_inequality :
  ∀ x : ℝ, f 2 x + f 2 (x/2 - 1) ≥ 5 ↔ x ≥ 3 ∨ x ≤ -1/3 :=
by sorry

end NUMINAMATH_CALUDE_find_a_solve_inequality_l938_93893


namespace NUMINAMATH_CALUDE_incorrect_arrangements_count_l938_93862

/-- The number of letters in the word --/
def word_length : ℕ := 4

/-- The total number of possible arrangements of the letters --/
def total_arrangements : ℕ := Nat.factorial word_length

/-- The number of correct arrangements (always 1 for a single word) --/
def correct_arrangements : ℕ := 1

/-- Theorem: The number of incorrect arrangements of a 4-letter word is 23 --/
theorem incorrect_arrangements_count :
  total_arrangements - correct_arrangements = 23 := by sorry

end NUMINAMATH_CALUDE_incorrect_arrangements_count_l938_93862


namespace NUMINAMATH_CALUDE_square_difference_equality_l938_93849

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l938_93849


namespace NUMINAMATH_CALUDE_some_number_value_l938_93853

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 25 * 35 * 63) : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l938_93853


namespace NUMINAMATH_CALUDE_remainder_sum_l938_93828

theorem remainder_sum (n : ℤ) : n % 20 = 11 → (n % 4 + n % 5 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l938_93828


namespace NUMINAMATH_CALUDE_distribute_five_items_three_bags_l938_93887

/-- The number of ways to distribute n distinct items into k identical bags,
    allowing for empty bags and the possibility of leaving items out. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 106 ways to distribute 5 distinct items
    into 3 identical bags, allowing for empty bags and the possibility of
    leaving one item out. -/
theorem distribute_five_items_three_bags : distribute 5 3 = 106 := by sorry

end NUMINAMATH_CALUDE_distribute_five_items_three_bags_l938_93887


namespace NUMINAMATH_CALUDE_greatest_three_digit_base_8_divisible_by_7_l938_93885

def base_8_to_decimal (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

def is_three_digit_base_8 (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_three_digit_base_8_divisible_by_7 :
  ∀ n : Nat, is_three_digit_base_8 n → (base_8_to_decimal n) % 7 = 0 →
  n ≤ 777 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_base_8_divisible_by_7_l938_93885


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l938_93871

-- Define the repeating decimal 0.343434...
def repeating_decimal : ℚ := 34 / 99

-- Theorem statement
theorem reciprocal_of_repeating_decimal :
  (repeating_decimal⁻¹ : ℚ) = 99 / 34 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l938_93871


namespace NUMINAMATH_CALUDE_circle_area_from_sector_l938_93818

theorem circle_area_from_sector (r : ℝ) (P : ℝ) (Q : ℝ) : 
  P = 2 → -- The area of sector COD is 2
  P = (1/6) * π * r^2 → -- Area of sector COD is 1/6 of circle area
  Q = π * r^2 → -- Q is the area of the entire circle
  Q = 12 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_sector_l938_93818


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l938_93895

/-- Given a circle D defined by the equation x^2 - 20x + y^2 + 6y + 25 = 0,
    prove that the sum of its center coordinates and radius is 7 + √66 -/
theorem circle_center_radius_sum :
  ∃ (c d s : ℝ),
    (∀ (x y : ℝ), x^2 - 20*x + y^2 + 6*y + 25 = 0 ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    c + d + s = 7 + Real.sqrt 66 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l938_93895


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l938_93855

-- Problem 1
theorem problem_1 : (1) - 1^2 + Real.sqrt 12 + Real.sqrt (4/3) = -1 + (8 * Real.sqrt 3) / 3 := by sorry

-- Problem 2
theorem problem_2 : ∀ x : ℝ, 2*x^2 - x - 1 = 0 ↔ x = -1/2 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l938_93855


namespace NUMINAMATH_CALUDE_exam_selection_difference_l938_93864

theorem exam_selection_difference (total_candidates : ℕ) 
  (selection_rate_A selection_rate_B : ℚ) : 
  total_candidates = 8200 →
  selection_rate_A = 6 / 100 →
  selection_rate_B = 7 / 100 →
  (selection_rate_B * total_candidates : ℚ).floor - 
  (selection_rate_A * total_candidates : ℚ).floor = 82 :=
by sorry

end NUMINAMATH_CALUDE_exam_selection_difference_l938_93864


namespace NUMINAMATH_CALUDE_valid_grid_count_l938_93863

/-- A type representing a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → Nat

/-- Checks if a grid is valid according to the problem rules -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j, j < 2 → g i j < g i (j+1)) ∧  -- rows in ascending order
  (∀ i j, i < 2 → g i j < g (i+1) j) ∧  -- columns in ascending order
  (∀ i j, g i j ∈ Finset.range 9) ∧     -- numbers from 1 to 9
  (g 0 0 = 1) ∧ (g 1 1 = 4) ∧ (g 2 2 = 9)  -- pre-filled numbers

/-- The set of all valid grids -/
def valid_grids : Finset Grid :=
  sorry

theorem valid_grid_count : Finset.card valid_grids = 12 := by
  sorry

end NUMINAMATH_CALUDE_valid_grid_count_l938_93863


namespace NUMINAMATH_CALUDE_frustum_cone_volume_l938_93816

/-- Given a frustum of a cone with volume 78 and one base area 9 times the other,
    the volume of the cone that cuts this frustum is 81. -/
theorem frustum_cone_volume (r R : ℝ) (h1 : r > 0) (h2 : R > 0) : 
  (π * (R^2 + r^2 + R*r) * (R - r) / 3 = 78) →
  (π * R^2 = 9 * π * r^2) →
  (π * R^3 / 3 = 81) := by
  sorry

end NUMINAMATH_CALUDE_frustum_cone_volume_l938_93816


namespace NUMINAMATH_CALUDE_direct_proportion_unique_k_l938_93813

/-- A function f: ℝ → ℝ is a direct proportion if there exists a non-zero constant m such that f(x) = m * x for all x -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ (m : ℝ), m ≠ 0 ∧ ∀ x, f x = m * x

/-- The function defined by k -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x + k^2 - 1

/-- Theorem stating that k = -1 is the only value that makes f a direct proportion function -/
theorem direct_proportion_unique_k :
  ∃! k, is_direct_proportion (f k) ∧ k = -1 :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_unique_k_l938_93813


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l938_93874

/-- Given a quadratic equation x^2 + 2 = 3x, prove that the coefficient of x^2 is 1 and the coefficient of x is -3. -/
theorem quadratic_equation_coefficients :
  let eq : ℝ → Prop := λ x => x^2 + 2 = 3*x
  ∃ a b c : ℝ, (∀ x, eq x ↔ a*x^2 + b*x + c = 0) ∧ a = 1 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l938_93874


namespace NUMINAMATH_CALUDE_set_operations_l938_93825

def A : Set ℤ := {x | -6 ≤ x ∧ x ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

theorem set_operations :
  (A ∪ (B ∩ C) = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6}) ∧
  (A ∩ (A \ (B ∩ C)) = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 4, 5, 6}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l938_93825


namespace NUMINAMATH_CALUDE_square_root_and_square_operations_l938_93852

theorem square_root_and_square_operations : 
  (∃ (x : ℝ), x ^ 2 = 4 ∧ x = 2) ∧ 
  (∀ (a : ℝ), (-3 * a) ^ 2 = 9 * a ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_square_root_and_square_operations_l938_93852


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l938_93898

theorem smallest_factor_for_perfect_square : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (m : ℕ), 1152 * n = m^2) ∧ 
  (∀ (k : ℕ), k > 0 → k < n → ¬∃ (m : ℕ), 1152 * k = m^2) ∧
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l938_93898


namespace NUMINAMATH_CALUDE_product_equals_32_l938_93846

theorem product_equals_32 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_32_l938_93846


namespace NUMINAMATH_CALUDE_division_problem_l938_93808

theorem division_problem (x y q : ℕ) : 
  y - x = 1375 →
  y = 1632 →
  y = q * x + 15 →
  q = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l938_93808


namespace NUMINAMATH_CALUDE_range_of_c_minus_b_l938_93841

/-- Represents a triangle with side lengths a, b, c and opposite angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The range of c - b in a triangle where a = 1 and C - B = π/2 -/
theorem range_of_c_minus_b (t : Triangle) 
  (h1 : t.a = 1) 
  (h2 : t.C - t.B = π/2) : 
  ∃ (l u : ℝ), l = Real.sqrt 2 / 2 ∧ u = 1 ∧ 
  ∀ x, (t.c - t.b = x) → l < x ∧ x < u :=
sorry

end NUMINAMATH_CALUDE_range_of_c_minus_b_l938_93841


namespace NUMINAMATH_CALUDE_second_number_proof_l938_93877

theorem second_number_proof (x : ℕ) : 
  (∃ k₁ k₂ : ℕ, 690 = 170 * k₁ + 10 ∧ x = 170 * k₂ + 25) ∧
  (∀ d : ℕ, d > 170 → ¬(∃ m₁ m₂ : ℕ, 690 = d * m₁ + 10 ∧ x = d * m₂ + 25)) →
  x = 875 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l938_93877


namespace NUMINAMATH_CALUDE_ned_remaining_games_l938_93860

/-- The number of games Ned initially had -/
def initial_games : ℕ := 19

/-- The number of games Ned gave away -/
def games_given_away : ℕ := 13

/-- The number of games Ned has now -/
def remaining_games : ℕ := initial_games - games_given_away

theorem ned_remaining_games : remaining_games = 6 := by
  sorry

end NUMINAMATH_CALUDE_ned_remaining_games_l938_93860


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l938_93819

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l938_93819


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l938_93834

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ = 129 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l938_93834


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l938_93881

theorem greatest_three_digit_number : ∃ n : ℕ,
  n = 989 ∧
  n < 1000 ∧
  ∃ k : ℕ, n = 7 * k + 2 ∧
  ∃ m : ℕ, n = 4 * m + 1 ∧
  ∀ x : ℕ, x < 1000 → (∃ a : ℕ, x = 7 * a + 2) → (∃ b : ℕ, x = 4 * b + 1) → x ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l938_93881


namespace NUMINAMATH_CALUDE_lottery_first_prize_probability_l938_93859

/-- The number of balls in the MegaBall drawing -/
def megaBallCount : ℕ := 30

/-- The number of balls in the WinnerBalls drawing -/
def winnerBallCount : ℕ := 50

/-- The number of WinnerBalls picked -/
def winnerBallsPicked : ℕ := 5

/-- The probability of winning the first prize in the lottery game -/
def firstPrizeProbability : ℚ := 1 / 127125600

/-- Theorem stating the probability of winning the first prize in the lottery game -/
theorem lottery_first_prize_probability :
  firstPrizeProbability = 1 / (megaBallCount * 2 * Nat.choose winnerBallCount winnerBallsPicked) :=
by sorry

end NUMINAMATH_CALUDE_lottery_first_prize_probability_l938_93859


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l938_93875

theorem inequality_system_solutions (m : ℝ) : 
  (∃ (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x + 5 > 0 ∧ x - m ≤ 1))) → 
  -3 ≤ m ∧ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l938_93875


namespace NUMINAMATH_CALUDE_derivative_of_even_function_l938_93894

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the condition that f(-x) = f(x) for all x
variable (h : ∀ x, f (-x) = f x)

-- Define g as the derivative of f
variable (g : ℝ → ℝ)
variable (hg : ∀ x, HasDerivAt f (g x) x)

-- State the theorem
theorem derivative_of_even_function :
  ∀ x, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_derivative_of_even_function_l938_93894


namespace NUMINAMATH_CALUDE_steve_reading_time_l938_93824

/-- Represents Steve's daily reading schedule in pages -/
def daily_reading : Fin 7 → ℕ
  | 0 => 100  -- Monday
  | 1 => 150  -- Tuesday
  | 2 => 100  -- Wednesday
  | 3 => 150  -- Thursday
  | 4 => 100  -- Friday
  | 5 => 50   -- Saturday
  | 6 => 0    -- Sunday

/-- The total number of pages in the book -/
def book_length : ℕ := 2100

/-- Calculate the total pages read in a week -/
def pages_per_week : ℕ := (List.range 7).map daily_reading |>.sum

/-- The number of weeks needed to read the book -/
def weeks_to_read : ℕ := (book_length + pages_per_week - 1) / pages_per_week

theorem steve_reading_time :
  weeks_to_read = 4 := by sorry

end NUMINAMATH_CALUDE_steve_reading_time_l938_93824


namespace NUMINAMATH_CALUDE_cos_90_degrees_is_zero_l938_93805

theorem cos_90_degrees_is_zero :
  let cos_36 : ℝ := (1 + Real.sqrt 5) / 4
  let cos_54 : ℝ := (1 - Real.sqrt 5) / 4
  let sin_36 : ℝ := Real.sqrt (10 - 2 * Real.sqrt 5) / 4
  let sin_54 : ℝ := Real.sqrt (10 + 2 * Real.sqrt 5) / 4
  let cos_sum := cos_36 * cos_54 - sin_36 * sin_54
  cos_sum = 0 := by sorry

end NUMINAMATH_CALUDE_cos_90_degrees_is_zero_l938_93805


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l938_93890

/-- A geometric sequence with 8 terms -/
def GeometricSequence (a : Fin 8 → ℝ) (q : ℝ) : Prop :=
  ∀ i : Fin 7, a (i + 1) = a i * q

theorem geometric_sequence_inequality 
  (a : Fin 8 → ℝ) 
  (q : ℝ) 
  (h_geom : GeometricSequence a q)
  (h_pos : ∀ i, a i > 0)
  (h_q : q ≠ 1) :
  a 0 + a 7 > a 3 + a 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l938_93890


namespace NUMINAMATH_CALUDE_intersection_equals_N_l938_93809

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = x^2}
def N : Set ℝ := {y | ∃ x > 0, y = x + 2}

-- State the theorem
theorem intersection_equals_N : M ∩ N = N := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_N_l938_93809


namespace NUMINAMATH_CALUDE_average_of_combined_results_l938_93829

theorem average_of_combined_results :
  let n₁ : ℕ := 100
  let n₂ : ℕ := 75
  let avg₁ : ℚ := 45
  let avg₂ : ℚ := 65
  let total_sum : ℚ := n₁ * avg₁ + n₂ * avg₂
  let total_count : ℕ := n₁ + n₂
  total_sum / total_count = 9375 / 175 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l938_93829


namespace NUMINAMATH_CALUDE_ab_value_is_32_l938_93835

def is_distinct (a b c d e f : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def is_valid_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6

theorem ab_value_is_32 :
  ∃ (a b c d e f : ℕ),
    is_distinct a b c d e f ∧
    is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧
    is_valid_digit d ∧ is_valid_digit e ∧ is_valid_digit f ∧
    (10 * a + b) * ((10 * c + d) - e) + f = 2021 ∧
    10 * a + b = 32 :=
by sorry

end NUMINAMATH_CALUDE_ab_value_is_32_l938_93835


namespace NUMINAMATH_CALUDE_price_problem_solution_l938_93857

/-- The price of sugar and salt -/
def price_problem (sugar_price salt_price : ℝ) : Prop :=
  let sugar_3kg_salt_1kg := 3 * sugar_price + salt_price
  sugar_price = 1.5 ∧ sugar_3kg_salt_1kg = 5 →
  2 * sugar_price + 5 * salt_price = 5.5

/-- The solution to the price problem -/
theorem price_problem_solution :
  ∃ (sugar_price salt_price : ℝ), price_problem sugar_price salt_price :=
sorry

end NUMINAMATH_CALUDE_price_problem_solution_l938_93857


namespace NUMINAMATH_CALUDE_computer_screen_height_l938_93880

theorem computer_screen_height (side : ℝ) (height : ℝ) : 
  side = 20 →
  height = 4 * side + 20 →
  height = 100 := by
sorry

end NUMINAMATH_CALUDE_computer_screen_height_l938_93880


namespace NUMINAMATH_CALUDE_tucker_tissues_left_l938_93884

/-- The number of tissues left after buying boxes and using some. -/
def tissues_left (tissues_per_box : ℕ) (boxes_bought : ℕ) (tissues_used : ℕ) : ℕ :=
  tissues_per_box * boxes_bought - tissues_used

/-- Theorem: Given 160 tissues per box, 3 boxes bought, and 210 tissues used, 270 tissues are left. -/
theorem tucker_tissues_left :
  tissues_left 160 3 210 = 270 := by
  sorry

end NUMINAMATH_CALUDE_tucker_tissues_left_l938_93884


namespace NUMINAMATH_CALUDE_tangent_product_equals_two_l938_93888

theorem tangent_product_equals_two :
  (1 + Real.tan (20 * π / 180)) * (1 + Real.tan (25 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_equals_two_l938_93888


namespace NUMINAMATH_CALUDE_sum_diff_parity_l938_93812

theorem sum_diff_parity (a b : ℤ) : Even (a + b) ↔ Even (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sum_diff_parity_l938_93812


namespace NUMINAMATH_CALUDE_solve_linear_equation_l938_93831

theorem solve_linear_equation (x : ℝ) :
  3 * x - 8 = 4 * x + 5 → x = -13 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l938_93831


namespace NUMINAMATH_CALUDE_initial_apples_count_l938_93883

/-- The number of apples Rachel picked from the tree -/
def apples_picked : ℕ := 4

/-- The number of apples remaining on the tree -/
def apples_remaining : ℕ := 3

/-- The initial number of apples on the tree -/
def initial_apples : ℕ := apples_picked + apples_remaining

theorem initial_apples_count : initial_apples = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_count_l938_93883


namespace NUMINAMATH_CALUDE_product_range_difference_l938_93858

theorem product_range_difference (f g : ℝ → ℝ) :
  (∀ x, -3 ≤ f x ∧ f x ≤ 9) →
  (∀ x, -1 ≤ g x ∧ g x ≤ 6) →
  (∃ a b, ∀ x, f x * g x ≤ a ∧ b ≤ f x * g x ∧ a - b = 72) :=
by sorry

end NUMINAMATH_CALUDE_product_range_difference_l938_93858


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_two_l938_93868

/-- Given a function f(x) = (x * e^x) / (e^(ax) - 1), prove that if f is even, then a = 2 -/
theorem even_function_implies_a_equals_two (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (x * Real.exp x) / (Real.exp (a * x) - 1) = 
    (-x * Real.exp (-x)) / (Real.exp (-a * x) - 1)) →
  a = 2 := by
sorry


end NUMINAMATH_CALUDE_even_function_implies_a_equals_two_l938_93868


namespace NUMINAMATH_CALUDE_cody_dumplings_l938_93878

def dumplings_problem (first_batch second_batch eaten_first shared_first shared_second additional_eaten : ℕ) : Prop :=
  let remaining_first := first_batch - eaten_first - shared_first
  let remaining_second := second_batch - shared_second
  let total_remaining := remaining_first + remaining_second - additional_eaten
  total_remaining = 10

theorem cody_dumplings :
  dumplings_problem 14 20 7 5 8 4 := by sorry

end NUMINAMATH_CALUDE_cody_dumplings_l938_93878


namespace NUMINAMATH_CALUDE_minutes_from_2222_to_midnight_l938_93836

def minutes_until_midnight (hour : Nat) (minute : Nat) : Nat :=
  (23 - hour) * 60 + (60 - minute)

theorem minutes_from_2222_to_midnight :
  minutes_until_midnight 22 22 = 98 := by
  sorry

end NUMINAMATH_CALUDE_minutes_from_2222_to_midnight_l938_93836


namespace NUMINAMATH_CALUDE_binomial_square_constant_l938_93822

/-- If 4x^2 + 12x + a is the square of a binomial, then a = 9 -/
theorem binomial_square_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 4*x^2 + 12*x + a = (2*x + b)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l938_93822


namespace NUMINAMATH_CALUDE_hyperbola_equation_l938_93848

/-- Given a hyperbola with the following properties:
  - Standard form equation: x²/a² - y²/b² = 1
  - a > 0 and b > 0
  - A focus at (2, 0)
  - Asymptotes: y = ±√3x
  Prove that the equation of the hyperbola is x² - y²/3 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (focus : (2 : ℝ) = (a^2 + b^2).sqrt)
  (asymptote : b/a = Real.sqrt 3) :
  ∀ x y : ℝ, x^2 - y^2/3 = 1 ↔ x^2/a^2 - y^2/b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l938_93848


namespace NUMINAMATH_CALUDE_prayer_difference_l938_93873

/-- Represents the number of prayers for a pastor in a week -/
structure WeeklyPrayers where
  regularDays : ℕ
  sunday : ℕ

/-- Calculates the total number of prayers in a week -/
def totalPrayers (wp : WeeklyPrayers) : ℕ :=
  wp.regularDays * 6 + wp.sunday

/-- Pastor Paul's prayer schedule -/
def paulPrayers : WeeklyPrayers where
  regularDays := 20
  sunday := 40

/-- Pastor Bruce's prayer schedule -/
def brucePrayers : WeeklyPrayers where
  regularDays := paulPrayers.regularDays / 2
  sunday := paulPrayers.sunday * 2

theorem prayer_difference : 
  totalPrayers paulPrayers - totalPrayers brucePrayers = 20 := by
  sorry


end NUMINAMATH_CALUDE_prayer_difference_l938_93873


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l938_93899

theorem cubic_roots_sum (a b c : ℝ) : 
  (3 * a^3 - 6 * a^2 + 99 * a - 2 = 0) →
  (3 * b^3 - 6 * b^2 + 99 * b - 2 = 0) →
  (3 * c^3 - 6 * c^2 + 99 * c - 2 = 0) →
  (a + b - 2)^3 + (b + c - 2)^3 + (c + a - 2)^3 = -196 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l938_93899


namespace NUMINAMATH_CALUDE_exists_point_with_no_interior_lattice_points_l938_93844

-- Define a point with integer coordinates
structure IntPoint where
  x : Int
  y : Int

-- Define a function to check if a point is on a line
def onLine (p : IntPoint) (a b c : Int) : Prop :=
  a * p.x + b * p.y = c

-- Define a function to check if a point is in the interior of a segment
def inInterior (p q r : IntPoint) : Prop :=
  ∃ t : Rat, 0 < t ∧ t < 1 ∧
  p.x = q.x + t * (r.x - q.x) ∧
  p.y = q.y + t * (r.y - q.y)

-- Main theorem
theorem exists_point_with_no_interior_lattice_points
  (A B C : IntPoint) (hABC : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  ∃ P : IntPoint,
    P ≠ A ∧ P ≠ B ∧ P ≠ C ∧
    (∀ Q : IntPoint, ¬(inInterior Q P A ∨ inInterior Q P B ∨ inInterior Q P C)) :=
  sorry

end NUMINAMATH_CALUDE_exists_point_with_no_interior_lattice_points_l938_93844


namespace NUMINAMATH_CALUDE_male_wage_is_35_l938_93823

/-- Represents the daily wage structure and worker composition of a building contractor -/
structure ContractorData where
  male_workers : ℕ
  female_workers : ℕ
  child_workers : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the daily wage of a male worker given the contractor's data -/
def male_wage (data : ContractorData) : ℕ :=
  let total_workers := data.male_workers + data.female_workers + data.child_workers
  let total_wage := total_workers * data.average_wage
  let female_total := data.female_workers * data.female_wage
  let child_total := data.child_workers * data.child_wage
  (total_wage - female_total - child_total) / data.male_workers

/-- Theorem stating that for the given contractor data, the male wage is 35 -/
theorem male_wage_is_35 (data : ContractorData) 
  (h1 : data.male_workers = 20)
  (h2 : data.female_workers = 15)
  (h3 : data.child_workers = 5)
  (h4 : data.female_wage = 20)
  (h5 : data.child_wage = 8)
  (h6 : data.average_wage = 26) :
  male_wage data = 35 := by
  sorry

#eval male_wage { 
  male_workers := 20, 
  female_workers := 15, 
  child_workers := 5, 
  female_wage := 20, 
  child_wage := 8, 
  average_wage := 26 
}

end NUMINAMATH_CALUDE_male_wage_is_35_l938_93823


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l938_93801

theorem right_triangle_side_length (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) (h5 : c = 17) (h6 : a = 15) : b = 8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l938_93801


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l938_93842

theorem complex_sum_of_parts (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  zi = Complex.mk 1 (-2) → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l938_93842


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l938_93870

theorem rectangular_box_volume 
  (face_area1 face_area2 face_area3 : ℝ) 
  (h1 : face_area1 = 18)
  (h2 : face_area2 = 50)
  (h3 : face_area3 = 45) :
  ∃ (l w h : ℝ), 
    l * w = face_area1 ∧ 
    w * h = face_area2 ∧ 
    l * h = face_area3 ∧ 
    l * w * h = 30 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l938_93870


namespace NUMINAMATH_CALUDE_jackson_courtyard_tile_cost_l938_93833

/-- Calculates the total cost of tiles for a courtyard -/
def total_tile_cost (length width : ℝ) (tiles_per_sqft : ℝ) (green_tile_percent : ℝ) (green_tile_cost red_tile_cost : ℝ) : ℝ :=
  let area := length * width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := green_tile_percent * total_tiles
  let red_tiles := total_tiles - green_tiles
  green_tiles * green_tile_cost + red_tiles * red_tile_cost

/-- Theorem stating the total cost of tiles for Jackson's courtyard -/
theorem jackson_courtyard_tile_cost :
  total_tile_cost 10 25 4 0.4 3 1.5 = 2100 :=
by sorry

end NUMINAMATH_CALUDE_jackson_courtyard_tile_cost_l938_93833


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l938_93867

theorem average_of_three_numbers : 
  let x : ℤ := -63
  let numbers : List ℤ := [2, 76, x]
  (numbers.sum : ℚ) / numbers.length = 5 := by sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l938_93867


namespace NUMINAMATH_CALUDE_rubber_elongation_improvement_l938_93865

def n : ℕ := 10

def z_bar : ℝ := 11

def s_squared : ℝ := 61

def significant_improvement (z_bar s_squared : ℝ) : Prop :=
  z_bar ≥ 2 * Real.sqrt (s_squared / n)

theorem rubber_elongation_improvement :
  significant_improvement z_bar s_squared :=
sorry

end NUMINAMATH_CALUDE_rubber_elongation_improvement_l938_93865


namespace NUMINAMATH_CALUDE_candy_probability_theorem_l938_93856

-- Define the type for a packet of candies
structure Packet where
  blue : ℕ
  total : ℕ

-- Define the function to calculate the probability of drawing a blue candy from a box
def boxProbability (p1 p2 : Packet) : ℚ :=
  (p1.blue + p2.blue : ℚ) / (p1.total + p2.total : ℚ)

-- Theorem statement
theorem candy_probability_theorem :
  ∃ (p1 p2 p3 p4 : Packet),
    (boxProbability p1 p2 = 5/13 ∨ boxProbability p1 p2 = 7/18) ∧
    (boxProbability p3 p4 ≠ 17/40) ∧
    (∀ (p5 p6 : Packet), 3/8 ≤ boxProbability p5 p6 ∧ boxProbability p5 p6 ≤ 2/5) :=
by sorry

end NUMINAMATH_CALUDE_candy_probability_theorem_l938_93856


namespace NUMINAMATH_CALUDE_distinguishable_cube_colorings_count_l938_93840

/-- The number of distinguishable ways to color a cube with six different colors -/
def distinguishable_cube_colorings : ℕ := 30

/-- A cube has six faces -/
def cube_faces : ℕ := 6

/-- The number of rotational symmetries of a cube -/
def cube_rotational_symmetries : ℕ := 24

/-- The total number of ways to arrange 6 colors on 6 faces -/
def total_arrangements : ℕ := 720  -- 6!

theorem distinguishable_cube_colorings_count :
  distinguishable_cube_colorings = total_arrangements / cube_rotational_symmetries :=
by sorry

end NUMINAMATH_CALUDE_distinguishable_cube_colorings_count_l938_93840


namespace NUMINAMATH_CALUDE_sqrt_two_difference_product_l938_93889

theorem sqrt_two_difference_product : (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_difference_product_l938_93889


namespace NUMINAMATH_CALUDE_total_arms_collected_l938_93814

theorem total_arms_collected (starfish_count : ℕ) (starfish_arms : ℕ) (seastar_count : ℕ) (seastar_arms : ℕ) :
  starfish_count = 7 →
  starfish_arms = 5 →
  seastar_count = 1 →
  seastar_arms = 14 →
  starfish_count * starfish_arms + seastar_count * seastar_arms = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_total_arms_collected_l938_93814


namespace NUMINAMATH_CALUDE_rainfall_difference_l938_93896

/-- The difference in rainfall between March and April -/
theorem rainfall_difference (march_rainfall april_rainfall : ℝ) 
  (h1 : march_rainfall = 0.81)
  (h2 : april_rainfall = 0.46) : 
  march_rainfall - april_rainfall = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_difference_l938_93896


namespace NUMINAMATH_CALUDE_compare_fractions_l938_93803

theorem compare_fractions : -8 / 21 > -3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l938_93803
