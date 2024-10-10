import Mathlib

namespace solution_difference_l3621_362182

-- Define the equation
def equation (x : ℝ) : Prop :=
  x ≠ 3 ∧ (2 * x^2 - 5 * x - 31) / (x - 3) = 3 * x + 8

-- Define the set of solutions
def solutions : Set ℝ :=
  {x | equation x}

-- State the theorem
theorem solution_difference : 
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 2 * Real.sqrt 11 :=
sorry

end solution_difference_l3621_362182


namespace hat_cost_l3621_362142

/-- The cost of each hat when a person has enough hats for 2 weeks and the total cost is $700 -/
theorem hat_cost (num_weeks : ℕ) (days_per_week : ℕ) (total_cost : ℕ) : 
  num_weeks = 2 → days_per_week = 7 → total_cost = 700 → 
  total_cost / (num_weeks * days_per_week) = 50 := by
  sorry

end hat_cost_l3621_362142


namespace conference_left_handed_fraction_l3621_362157

theorem conference_left_handed_fraction 
  (total : ℕ) 
  (red : ℕ) 
  (blue : ℕ) 
  (h1 : red + blue = total) 
  (h2 : red = 2 * blue) 
  (h3 : red > 0) 
  (h4 : blue > 0) : 
  (red * (1/3) + blue * (2/3)) / total = 4/9 := by
  sorry

end conference_left_handed_fraction_l3621_362157


namespace juice_price_ratio_l3621_362135

theorem juice_price_ratio :
  let volume_A : ℝ := 1.25  -- Brand A's volume relative to Brand B
  let price_A : ℝ := 0.85   -- Brand A's price relative to Brand B
  let unit_price_ratio := (price_A / volume_A) / 1  -- Ratio of unit prices (A / B)
  unit_price_ratio = 17 / 25 := by
  sorry

end juice_price_ratio_l3621_362135


namespace fencing_cost_distribution_impossible_equal_distribution_impossible_l3621_362116

/-- Represents the dimensions of the cottage settlement. -/
structure Settlement where
  n : ℕ
  m : ℕ

/-- Calculates the total cost of fencing for the entire settlement. -/
def totalFencingCost (s : Settlement) : ℕ :=
  10000 * (2 * s.n * s.m + s.n + s.m - 4)

/-- Calculates the sum of costs if equal numbers of residents spent 0, 10000, 30000, 40000,
    and the rest spent 20000 rubles. -/
def proposedCostSum (s : Settlement) : ℕ :=
  100000 + 20000 * (s.n * s.m - 4)

/-- Theorem stating that the proposed cost distribution is impossible. -/
theorem fencing_cost_distribution_impossible (s : Settlement) :
  totalFencingCost s ≠ proposedCostSum s :=
sorry

/-- Theorem stating that it's impossible to have equal numbers of residents spending
    0, 10000, 30000, 40000 rubles with the rest spending 20000 rubles. -/
theorem equal_distribution_impossible (s : Settlement) :
  ¬ ∃ (k : ℕ), k > 0 ∧ 
    s.n * s.m = 4 * k + (s.n * s.m - 4 * k) ∧
    totalFencingCost s = k * (0 + 10000 + 30000 + 40000) + (s.n * s.m - 4 * k) * 20000 :=
sorry

end fencing_cost_distribution_impossible_equal_distribution_impossible_l3621_362116


namespace integer_divisibility_problem_l3621_362110

theorem integer_divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (((a - 1) * (b - 1) * (c - 1)) ∣ (a * b * c - 1)) →
    ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by sorry

end integer_divisibility_problem_l3621_362110


namespace count_four_digit_distinct_prime_last_l3621_362199

/-- A function that returns true if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The set of single-digit prime numbers -/
def singleDigitPrimes : Finset ℕ := sorry

/-- A function that returns the set of digits of a natural number -/
def digits (n : ℕ) : Finset ℕ := sorry

/-- The count of four-digit numbers with distinct digits and a prime last digit -/
def countValidNumbers : ℕ := sorry

theorem count_four_digit_distinct_prime_last :
  countValidNumbers = 1344 := by
  sorry

end count_four_digit_distinct_prime_last_l3621_362199


namespace second_grade_sample_l3621_362134

/-- Given a total sample size and ratios for three grades, 
    calculate the number of students to be drawn from a specific grade. -/
def stratified_sample (total_sample : ℕ) (ratio1 ratio2 ratio3 : ℕ) (target_ratio : ℕ) : ℕ :=
  (target_ratio * total_sample) / (ratio1 + ratio2 + ratio3)

/-- Theorem: Given a total sample of 50 and ratios 3:3:4, 
    the number of students from the grade with ratio 3 is 15. -/
theorem second_grade_sample :
  stratified_sample 50 3 3 4 3 = 15 := by
  sorry

end second_grade_sample_l3621_362134


namespace max_missed_questions_correct_l3621_362107

/-- The number of questions in the test -/
def total_questions : ℕ := 50

/-- The minimum passing percentage -/
def passing_percentage : ℚ := 85 / 100

/-- The greatest number of questions a student can miss and still pass -/
def max_missed_questions : ℕ := 7

theorem max_missed_questions_correct :
  max_missed_questions = ⌊(1 - passing_percentage) * total_questions⌋ := by
  sorry

end max_missed_questions_correct_l3621_362107


namespace mirror_area_l3621_362113

theorem mirror_area (frame_width frame_height frame_thickness : ℕ) : 
  frame_width = 100 ∧ frame_height = 120 ∧ frame_thickness = 15 →
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 6300 :=
by sorry

end mirror_area_l3621_362113


namespace simplify_x_expression_simplify_a_expression_l3621_362170

-- First equation
theorem simplify_x_expression (x : ℝ) : 3 * x^4 * x^2 + (2 * x^2)^3 = 11 * x^6 := by
  sorry

-- Second equation
theorem simplify_a_expression (a : ℝ) : 3 * a * (9 * a + 3) - 4 * a * (2 * a - 1) = 19 * a^2 + 13 * a := by
  sorry

end simplify_x_expression_simplify_a_expression_l3621_362170


namespace distance_center_to_point_l3621_362122

/-- Given a circle with polar equation ρ = 4cosθ and a point P with polar coordinates (4, π/3),
    prove that the distance between the center of the circle and point P is 2√3. -/
theorem distance_center_to_point (θ : Real) (ρ : Real → Real) (P : Real × Real) :
  (ρ = fun θ => 4 * Real.cos θ) →
  P = (4, Real.pi / 3) →
  ∃ C : Real × Real, 
    (C.1 - P.1)^2 + (C.2 - P.2)^2 = 12 :=
by sorry

end distance_center_to_point_l3621_362122


namespace willys_work_problem_l3621_362123

/-- Willy's work problem -/
theorem willys_work_problem (total_days : ℕ) (daily_wage : ℚ) (daily_fine : ℚ) 
  (h_total_days : total_days = 30)
  (h_daily_wage : daily_wage = 8)
  (h_daily_fine : daily_fine = 10)
  (h_no_money_owed : ∃ (days_worked : ℚ), 
    0 ≤ days_worked ∧ 
    days_worked ≤ total_days ∧ 
    days_worked * daily_wage = (total_days - days_worked) * daily_fine) :
  ∃ (days_worked : ℚ) (days_missed : ℚ),
    days_worked = 50 / 3 ∧
    days_missed = 40 / 3 ∧
    days_worked + days_missed = total_days ∧
    days_worked * daily_wage = days_missed * daily_fine :=
sorry

end willys_work_problem_l3621_362123


namespace quadratic_inequality_range_l3621_362196

theorem quadratic_inequality_range (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 2 4 ∧ x^2 - 2*x + 5 - m < 0) ↔ m > 13 :=
sorry

end quadratic_inequality_range_l3621_362196


namespace inequalities_proof_l3621_362197

theorem inequalities_proof (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) : 
  (a + c > b + d) ∧ (a * d^2 > b * c^2) ∧ (1 / (b * c) < 1 / (a * d)) := by
  sorry

end inequalities_proof_l3621_362197


namespace arithmetic_sequence_general_term_l3621_362167

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_10 : a 10 = 30)
  (h_20 : a 20 = 50) :
  ∃ b c : ℝ, ∀ n : ℕ, a n = b * n + c ∧ b = 2 ∧ c = 10 :=
sorry

end arithmetic_sequence_general_term_l3621_362167


namespace workshop_problem_l3621_362148

theorem workshop_problem :
  ∃ (x y : ℕ),
    x ≥ 1 ∧ y ≥ 1 ∧
    6 + 11 * (x - 1) = 7 + 10 * (y - 1) ∧
    100 ≤ 6 + 11 * (x - 1) ∧
    6 + 11 * (x - 1) ≤ 200 ∧
    x = 12 ∧ y = 13 :=
by sorry

end workshop_problem_l3621_362148


namespace hyperbola_eccentricity_range_l3621_362102

/-- Given a hyperbola C: (y^2 / a^2) - (x^2 / b^2) = 1 with a > 0 and b > 0,
    whose asymptotes intersect with the circle x^2 + (y - 2)^2 = 1,
    the eccentricity e of C satisfies 1 < e < 2√3/3. -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | y^2 / a^2 - x^2 / b^2 = 1}
  let asymptotes := {(x, y) : ℝ × ℝ | a * x = b * y ∨ a * x = -b * y}
  let circle := {(x, y) : ℝ × ℝ | x^2 + (y - 2)^2 = 1}
  let e := Real.sqrt (1 + b^2 / a^2)
  (∃ p ∈ asymptotes, p ∈ circle) →
  1 < e ∧ e < 2 * Real.sqrt 3 / 3 :=
by sorry

end hyperbola_eccentricity_range_l3621_362102


namespace line_parallel_to_intersection_l3621_362130

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallelLine : Line → Line → Prop)

-- Define the intersection of two planes resulting in a line
variable (planeIntersection : Plane → Plane → Line)

-- Theorem statement
theorem line_parallel_to_intersection
  (l m : Line) (α β : Plane)
  (h1 : planeIntersection α β = l)
  (h2 : parallelLinePlane m α)
  (h3 : parallelLinePlane m β) :
  parallelLine m l :=
sorry

end line_parallel_to_intersection_l3621_362130


namespace quadratic_complete_square_l3621_362100

theorem quadratic_complete_square (x : ℝ) : 
  x^2 + 10*x + 7 = 0 → ∃ c d : ℝ, (x + c)^2 = d ∧ d = 18 :=
by sorry

end quadratic_complete_square_l3621_362100


namespace least_marbles_thirty_two_satisfies_george_marbles_l3621_362168

theorem least_marbles (n : ℕ) : 
  (n % 7 = 1 ∧ n % 4 = 2 ∧ n % 6 = 3) → n ≥ 32 :=
by sorry

theorem thirty_two_satisfies : 
  32 % 7 = 1 ∧ 32 % 4 = 2 ∧ 32 % 6 = 3 :=
by sorry

theorem george_marbles : 
  ∃ (n : ℕ), n % 7 = 1 ∧ n % 4 = 2 ∧ n % 6 = 3 ∧ 
  ∀ (m : ℕ), (m % 7 = 1 ∧ m % 4 = 2 ∧ m % 6 = 3) → n ≤ m :=
by sorry

end least_marbles_thirty_two_satisfies_george_marbles_l3621_362168


namespace probability_exactly_two_eights_value_l3621_362119

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def target_value : ℕ := 8
def num_target : ℕ := 2

def probability_exactly_two_eights : ℚ :=
  (Nat.choose num_dice num_target) *
  (1 / num_sides) ^ num_target *
  ((num_sides - 1) / num_sides) ^ (num_dice - num_target)

theorem probability_exactly_two_eights_value :
  probability_exactly_two_eights = 28 * 117649 / 16777216 :=
sorry

end probability_exactly_two_eights_value_l3621_362119


namespace not_p_or_not_q_false_implies_l3621_362128

theorem not_p_or_not_q_false_implies (p q : Prop) 
  (h : ¬(¬p ∨ ¬q)) : 
  (p ∧ q) ∧ (p ∨ q) := by
  sorry

end not_p_or_not_q_false_implies_l3621_362128


namespace exists_selling_price_with_50_percent_profit_l3621_362176

/-- Represents the pricing model for a printer -/
structure PricingModel where
  baseSellPrice : ℝ
  baseProfit : ℝ
  taxRate1 : ℝ
  taxRate2 : ℝ
  taxThreshold1 : ℝ
  taxThreshold2 : ℝ
  discountRate : ℝ
  discountIncrement : ℝ

/-- Calculates the selling price that yields the target profit percentage -/
def findSellingPrice (model : PricingModel) (targetProfit : ℝ) : ℝ :=
  sorry

/-- Theorem: There exists a selling price that yields a 50% profit on the cost of the printer -/
theorem exists_selling_price_with_50_percent_profit (model : PricingModel) :
  ∃ (sellPrice : ℝ), findSellingPrice model 0.5 = sellPrice :=
by
  sorry

end exists_selling_price_with_50_percent_profit_l3621_362176


namespace sqrt_five_power_calculation_l3621_362158

theorem sqrt_five_power_calculation :
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 78125 * Real.sqrt 5 := by
  sorry

end sqrt_five_power_calculation_l3621_362158


namespace negative_64_to_7_6th_l3621_362152

theorem negative_64_to_7_6th : ∃ (z : ℂ), z^6 = (-64)^7 ∧ z = 128 * Complex.I :=
by
  sorry

end negative_64_to_7_6th_l3621_362152


namespace triangle_inequality_l3621_362154

/-- 
Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
if a^2 - √3ab + b^2 = 1 and c = 1, then 1 < √3a - b < √3.
-/
theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  a^2 - Real.sqrt 3 * a * b + b^2 = 1 ∧  -- Given condition
  c = 1 →  -- Given condition
  1 < Real.sqrt 3 * a - b ∧ Real.sqrt 3 * a - b < Real.sqrt 3 := by
sorry

end triangle_inequality_l3621_362154


namespace inequality_proof_l3621_362115

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b) > 0 := by
  sorry

end inequality_proof_l3621_362115


namespace max_l_value_l3621_362151

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 8 * x + 3

-- Define the condition for l(a)
def is_valid_l (a : ℝ) (l : ℝ) : Prop :=
  a < 0 ∧ l > 0 ∧ ∀ x ∈ Set.Icc 0 l, |f a x| ≤ 5

-- Define l(a) as the supremum of valid l values
noncomputable def l (a : ℝ) : ℝ :=
  ⨆ (l : ℝ) (h : is_valid_l a l), l

-- State the theorem
theorem max_l_value :
  ∃ (a : ℝ), a < 0 ∧ l a = (Real.sqrt 5 + 1) / 2 ∧
  ∀ (b : ℝ), b < 0 → l b ≤ l a :=
sorry

end max_l_value_l3621_362151


namespace intersection_of_M_and_N_l3621_362109

/-- Given sets M and N, prove their intersection -/
theorem intersection_of_M_and_N :
  let M := {x : ℝ | |x - 1| < 2}
  let N := {x : ℝ | x * (x - 3) < 0}
  M ∩ N = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end intersection_of_M_and_N_l3621_362109


namespace solve_equation_l3621_362175

theorem solve_equation : 
  ∃ x : ℝ, (4.7 * x + 4.7 * 9.43 + 4.7 * 77.31 = 470) ∧ (x = 13.26) :=
by
  sorry

end solve_equation_l3621_362175


namespace rolling_circle_trajectory_is_line_segment_l3621_362174

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point on a circle -/
structure PointOnCircle where
  circle : Circle
  angle : ℝ  -- Angle from the positive x-axis

/-- Represents the trajectory of a point -/
inductive Trajectory
  | LineSegment : (ℝ × ℝ) → (ℝ × ℝ) → Trajectory

/-- The trajectory of a fixed point on a circle rolling inside another circle -/
def rollingCircleTrajectory (stationaryCircle : Circle) (movingCircle : Circle) (fixedPoint : PointOnCircle) : Trajectory :=
  sorry

/-- Main theorem: The trajectory of a fixed point on a circle rolling inside another circle with twice its radius is a line segment -/
theorem rolling_circle_trajectory_is_line_segment
  (stationaryCircle : Circle)
  (movingCircle : Circle)
  (fixedPoint : PointOnCircle)
  (h1 : movingCircle.radius = stationaryCircle.radius / 2)
  (h2 : fixedPoint.circle = movingCircle) :
  ∃ (p q : ℝ × ℝ), rollingCircleTrajectory stationaryCircle movingCircle fixedPoint = Trajectory.LineSegment p q ∧
                    p = stationaryCircle.center :=
  sorry

end rolling_circle_trajectory_is_line_segment_l3621_362174


namespace arithmetic_sequence_sum_l3621_362118

/-- 
Given an arithmetic sequence of consecutive integers where:
- k is a natural number
- The first term is k^2 - 1
- The number of terms is 2k - 1

The sum of all terms in this sequence is equal to 2k^3 + k^2 - 5k + 2
-/
theorem arithmetic_sequence_sum (k : ℕ) : 
  let first_term := k^2 - 1
  let num_terms := 2*k - 1
  let last_term := first_term + (num_terms - 1)
  (num_terms : ℝ) * (first_term + last_term) / 2 = 2*k^3 + k^2 - 5*k + 2 :=
by sorry

end arithmetic_sequence_sum_l3621_362118


namespace complex_subtraction_simplification_l3621_362106

theorem complex_subtraction_simplification :
  (5 - 3*I) - (2 + 7*I) = 3 - 10*I :=
by sorry

end complex_subtraction_simplification_l3621_362106


namespace garden_area_l3621_362101

theorem garden_area (length_distance width_distance : ℝ) 
  (h1 : length_distance * 30 = 1500)
  (h2 : (2 * length_distance + 2 * width_distance) * 12 = 1500) :
  length_distance * width_distance = 625 := by
  sorry

end garden_area_l3621_362101


namespace inequality_proof_l3621_362121

theorem inequality_proof (x y : ℝ) (h1 : x > y) (h2 : x * y = 1) :
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := by
  sorry

end inequality_proof_l3621_362121


namespace square_width_proof_l3621_362188

theorem square_width_proof (rectangle_length : ℝ) (rectangle_width : ℝ) (area_difference : ℝ) :
  rectangle_length = 3 →
  rectangle_width = 6 →
  area_difference = 7 →
  ∃ (square_width : ℝ), square_width^2 = rectangle_length * rectangle_width - area_difference :=
by
  sorry

end square_width_proof_l3621_362188


namespace geometric_sequence_sum_l3621_362138

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 4 + a 5 + a 6 = 168 := by
  sorry


end geometric_sequence_sum_l3621_362138


namespace garden_dimensions_and_walkway_area_l3621_362194

/-- A rectangular garden with a surrounding walkway. -/
structure Garden where
  breadth : ℝ
  length : ℝ
  walkwayWidth : ℝ

/-- Properties of the garden based on the problem conditions. -/
def GardenProperties (g : Garden) : Prop :=
  g.length = 3 * g.breadth ∧
  2 * (g.length + g.breadth) = 40 ∧
  g.walkwayWidth = 1 ∧
  (g.length + 2 * g.walkwayWidth) * (g.breadth + 2 * g.walkwayWidth) = 120

theorem garden_dimensions_and_walkway_area 
  (g : Garden) 
  (h : GardenProperties g) : 
  g.length = 15 ∧ g.breadth = 5 ∧ 
  ((g.length + 2 * g.walkwayWidth) * (g.breadth + 2 * g.walkwayWidth) - g.length * g.breadth) = 45 :=
by sorry

end garden_dimensions_and_walkway_area_l3621_362194


namespace diamond_symmetry_lines_l3621_362160

/-- Definition of the diamond operation -/
def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

/-- The set of points (x, y) where x ◊ y = y ◊ x forms four lines -/
theorem diamond_symmetry_lines :
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1} =
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2} :=
by sorry

end diamond_symmetry_lines_l3621_362160


namespace largest_s_value_l3621_362131

theorem largest_s_value (r s : ℕ) : 
  r ≥ s → 
  s ≥ 5 → 
  (r - 2) * s * 61 = (s - 2) * r * 60 → 
  s ≤ 121 ∧ ∃ r' : ℕ, r' ≥ 121 ∧ (r' - 2) * 121 * 61 = (121 - 2) * r' * 60 :=
by sorry

end largest_s_value_l3621_362131


namespace function_upper_bound_l3621_362149

theorem function_upper_bound
  (a r : ℝ)
  (ha : a > 1)
  (hr : r > 1)
  (f : ℝ → ℝ)
  (hf_pos : ∀ x > 0, f x > 0)
  (hf_cond1 : ∀ x > 0, (f x)^2 ≤ a * x^r * f (x/a))
  (hf_cond2 : ∀ x > 0, x < 1/2^2000 → f x < 2^2000) :
  ∀ x > 0, f x ≤ x^r * a^(1-r) := by
sorry

end function_upper_bound_l3621_362149


namespace train_length_l3621_362112

/-- The length of a train given its speed, time to pass a platform, and platform length. -/
theorem train_length (train_speed : ℝ) (time_to_pass : ℝ) (platform_length : ℝ) :
  train_speed = 60 →
  time_to_pass = 23.998080153587715 →
  platform_length = 260 →
  (train_speed * 1000 / 3600) * time_to_pass - platform_length = 140 := by
  sorry

#check train_length

end train_length_l3621_362112


namespace arctan_sum_three_seven_l3621_362150

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end arctan_sum_three_seven_l3621_362150


namespace candy_remaining_l3621_362172

theorem candy_remaining (initial : ℕ) (talitha solomon maya : ℕ) 
  (h1 : initial = 572)
  (h2 : talitha = 183)
  (h3 : solomon = 238)
  (h4 : maya = 127) :
  initial - (talitha + solomon + maya) = 24 :=
by sorry

end candy_remaining_l3621_362172


namespace flight_savings_l3621_362169

theorem flight_savings (delta_price united_price : ℝ) 
  (delta_discount united_discount : ℝ) :
  delta_price = 850 →
  united_price = 1100 →
  delta_discount = 0.20 →
  united_discount = 0.30 →
  united_price * (1 - united_discount) - delta_price * (1 - delta_discount) = 90 := by
  sorry

end flight_savings_l3621_362169


namespace max_sum_abc_l3621_362184

def An (a n : ℕ) : ℚ := a * (10^n - 1) / 9
def Bn (b n : ℕ) : ℚ := b * (10^(2*n) - 1) / 9
def Cn (c n : ℕ) : ℚ := c * (10^(2*n) - 1) / 9

theorem max_sum_abc (a b c n : ℕ) :
  (a ∈ Finset.range 10 ∧ a ≠ 0) →
  (b ∈ Finset.range 10 ∧ b ≠ 0) →
  (c ∈ Finset.range 10 ∧ c ≠ 0) →
  (∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ Cn c n₁ - Bn b n₁ = (An a n₁)^2 ∧ Cn c n₂ - Bn b n₂ = (An a n₂)^2) →
  a + b + c ≤ 18 :=
by sorry

end max_sum_abc_l3621_362184


namespace sqrt_equation_solution_l3621_362183

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (5 * x + 9) = 12) ∧ (x = 27) :=
by sorry

end sqrt_equation_solution_l3621_362183


namespace expected_successful_trials_value_l3621_362132

/-- A trial is successful if at least one of two dice shows a 4 or a 5 -/
def is_successful_trial (dice1 dice2 : Nat) : Bool :=
  dice1 = 4 ∨ dice1 = 5 ∨ dice2 = 4 ∨ dice2 = 5

/-- The probability of a successful trial -/
def prob_success : ℚ := 5 / 9

/-- The number of trials -/
def num_trials : ℕ := 10

/-- The expected number of successful trials -/
def expected_successful_trials : ℚ := num_trials * prob_success

theorem expected_successful_trials_value :
  expected_successful_trials = 50 / 9 := by sorry

end expected_successful_trials_value_l3621_362132


namespace smallest_number_with_17_proper_factors_l3621_362159

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

def number_of_proper_factors (n : ℕ) : ℕ := (number_of_factors n) - 2

theorem smallest_number_with_17_proper_factors :
  ∃ (n : ℕ), n > 0 ∧ 
    number_of_factors n = 19 ∧ 
    number_of_proper_factors n = 17 ∧
    ∀ (m : ℕ), m > 0 → number_of_factors m = 19 → m ≥ n :=
by
  -- The proof would go here
  sorry

end smallest_number_with_17_proper_factors_l3621_362159


namespace joels_board_games_l3621_362163

theorem joels_board_games (stuffed_animals action_figures puzzles total_toys joels_toys : ℕ)
  (h1 : stuffed_animals = 18)
  (h2 : action_figures = 42)
  (h3 : puzzles = 13)
  (h4 : total_toys = 108)
  (h5 : joels_toys = 22) :
  ∃ (board_games sisters_toys : ℕ),
    sisters_toys * 3 = joels_toys ∧
    stuffed_animals + action_figures + board_games + puzzles + sisters_toys * 3 = total_toys ∧
    board_games = 14 :=
by
  sorry

end joels_board_games_l3621_362163


namespace smallest_gcd_of_multiples_l3621_362143

theorem smallest_gcd_of_multiples (m n : ℕ+) (h : Nat.gcd m n = 15) :
  ∃ (k : ℕ), k ≥ 30 ∧ Nat.gcd (14 * m) (20 * n) = k ∧
  ∀ (j : ℕ), j < 30 → Nat.gcd (14 * m) (20 * n) ≠ j :=
by sorry

end smallest_gcd_of_multiples_l3621_362143


namespace unique_solution_equals_three_l3621_362181

theorem unique_solution_equals_three :
  ∃! (x : ℝ), (x^2 - t*x + 36 = 0) ∧ (x^2 - 8*x + t = 0) ∧ x = 3 :=
by sorry

end unique_solution_equals_three_l3621_362181


namespace ellipse_properties_l3621_362129

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0

-- Define the focal distance
def focal_distance (c : ℝ) : Prop := c = 2

-- Define the eccentricity relation
def eccentricity_relation (a b : ℝ) : Prop :=
  (2 / a)^2 = 1 / 2

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * x + 1

-- Define the intersection condition
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ 
    ellipse_C x₁ y₁ (2 * Real.sqrt 2) 2 ∧
    ellipse_C x₂ y₂ (2 * Real.sqrt 2) 2 ∧
    line_l x₁ y₁ k ∧ line_l x₂ y₂ k

-- Define the focus inside circle condition
def focus_inside_circle (k : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂, 
    ellipse_C x₁ y₁ (2 * Real.sqrt 2) 2 →
    ellipse_C x₂ y₂ (2 * Real.sqrt 2) 2 →
    line_l x₁ y₁ k → line_l x₂ y₂ k →
    (x₁ - 2) * (x₂ - 2) + y₁ * y₂ < 0

theorem ellipse_properties :
  ∀ a b : ℝ,
    ellipse_C 0 0 a b →
    focal_distance 2 →
    eccentricity_relation a b →
    (a = 2 * Real.sqrt 2 ∧ b = 2) ∧
    (∀ k : ℝ, intersects_at_two_points k →
      (focus_inside_circle k ↔ k < 1/8)) := by sorry

end ellipse_properties_l3621_362129


namespace definite_integral_problem_l3621_362177

open Real MeasureTheory Interval Set

theorem definite_integral_problem :
  ∫ x in Icc 0 π, (2 * x^2 + 4 * x + 7) * cos (2 * x) = π := by
  sorry

end definite_integral_problem_l3621_362177


namespace circus_ticket_cost_l3621_362137

/-- Calculates the total cost of circus tickets -/
def total_ticket_cost (adult_price children_price senior_price : ℕ) 
  (adult_count children_count senior_count : ℕ) : ℕ :=
  adult_price * adult_count + children_price * children_count + senior_price * senior_count

/-- Proves that the total cost of circus tickets for the given quantities and prices is $318 -/
theorem circus_ticket_cost : 
  total_ticket_cost 55 28 42 4 2 1 = 318 := by
  sorry

end circus_ticket_cost_l3621_362137


namespace parking_lot_vehicles_l3621_362108

/-- Given a parking lot with tricycles and bicycles, prove the number of each type --/
theorem parking_lot_vehicles (total_vehicles : ℕ) (total_wheels : ℕ) 
  (h1 : total_vehicles = 15)
  (h2 : total_wheels = 40) :
  ∃ (tricycles bicycles : ℕ),
    tricycles + bicycles = total_vehicles ∧
    3 * tricycles + 2 * bicycles = total_wheels ∧
    tricycles = 10 ∧
    bicycles = 5 := by
  sorry

end parking_lot_vehicles_l3621_362108


namespace fraction_equality_implies_numerator_equality_l3621_362179

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end fraction_equality_implies_numerator_equality_l3621_362179


namespace intersection_implies_a_equals_two_l3621_362162

def A (a : ℝ) : Set ℝ := {2, a^2 - a + 1}
def B (a : ℝ) : Set ℝ := {3, a + 3}

theorem intersection_implies_a_equals_two (a : ℝ) :
  A a ∩ B a = {3} → a = 2 := by
  sorry

end intersection_implies_a_equals_two_l3621_362162


namespace smallest_c_in_arithmetic_progression_l3621_362114

theorem smallest_c_in_arithmetic_progression (a b c d : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →
  (∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r) →
  a * b * c * d = 256 →
  ∀ x : ℝ, (∃ a' b' d' : ℝ, 
    0 < a' ∧ 0 < b' ∧ 0 < x ∧ 0 < d' ∧
    (∃ r' : ℝ, b' = a' + r' ∧ x = b' + r' ∧ d' = x + r') ∧
    a' * b' * x * d' = 256) →
  x ≥ 4 :=
sorry

end smallest_c_in_arithmetic_progression_l3621_362114


namespace extremum_implies_b_value_l3621_362171

/-- A function f with a real parameter a -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f with respect to x -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_b_value (a b : ℝ) :
  (f' a b 1 = 0) →  -- Derivative is zero at x = 1
  (f a b 1 = 10) →  -- Function value is 10 at x = 1
  b = -11 := by
sorry

end extremum_implies_b_value_l3621_362171


namespace rectangle_division_l3621_362198

theorem rectangle_division (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  let x := b / 3
  let y := a / 3
  (x * a) / ((b - x) * a) = 1 / 2 ∧
  (2 * a + 2 * x) / (2 * a + 2 * (b - x)) = 3 / 5 →
  (y * b) / ((a - y) * b) = 1 / 2 →
  (2 * y + 2 * b) / (2 * (a - y) + 2 * b) = 20 / 19 :=
by sorry


end rectangle_division_l3621_362198


namespace balls_per_package_l3621_362145

theorem balls_per_package (total_packages : Nat) (total_balls : Nat) 
  (h1 : total_packages = 21) 
  (h2 : total_balls = 399) : 
  (total_balls / total_packages : Nat) = 19 := by
  sorry

end balls_per_package_l3621_362145


namespace sunglasses_cap_probability_l3621_362125

theorem sunglasses_cap_probability (total_sunglasses : ℕ) (total_caps : ℕ) 
  (prob_sunglasses_given_cap : ℚ) :
  total_sunglasses = 70 →
  total_caps = 45 →
  prob_sunglasses_given_cap = 3/9 →
  (prob_sunglasses_given_cap * total_caps : ℚ) / total_sunglasses = 3/14 :=
by sorry

end sunglasses_cap_probability_l3621_362125


namespace shift_increasing_interval_l3621_362190

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem shift_increasing_interval :
  IncreasingOn f (-2) 3 → IncreasingOn (fun x ↦ f (x + 4)) (-6) (-1) := by
  sorry

end shift_increasing_interval_l3621_362190


namespace ratio_equality_l3621_362178

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc_sq : a^2 + b^2 + c^2 = 1)
  (sum_xyz_sq : x^2 + y^2 + z^2 = 4)
  (sum_prod : a*x + b*y + c*z = 2) :
  (a + b + c) / (x + y + z) = 1/2 := by
sorry

end ratio_equality_l3621_362178


namespace prime_cube_difference_equation_l3621_362186

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem prime_cube_difference_equation :
  ∀ p q r : ℕ,
  is_prime p ∧ is_prime q ∧ is_prime r →
  p^3 - q^3 = 5*r →
  p = 7 ∧ q = 2 ∧ r = 67 :=
sorry

end prime_cube_difference_equation_l3621_362186


namespace arithmetic_sequence_sum_property_l3621_362146

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- Sum function

/-- Theorem: For an arithmetic sequence, if S_p = q and S_q = p where p ≠ q, then S_{p+q} = -(p + q) -/
theorem arithmetic_sequence_sum_property (a : ArithmeticSequence) (p q : ℕ) 
    (h1 : a.S p = q)
    (h2 : a.S q = p)
    (h3 : p ≠ q) : 
  a.S (p + q) = -(p + q) := by
  sorry

end arithmetic_sequence_sum_property_l3621_362146


namespace min_diagonal_pairs_l3621_362126

/-- Represents a triangle of cells arranged in rows -/
structure CellTriangle where
  rows : ℕ

/-- Calculates the total number of cells in the triangle -/
def totalCells (t : CellTriangle) : ℕ :=
  t.rows * (t.rows + 1) / 2

/-- Calculates the number of rows with an odd number of cells -/
def oddRows (t : CellTriangle) : ℕ :=
  t.rows / 2

/-- Theorem: The minimum number of diagonal pairs in a cell triangle
    with 5784 rows is equal to the number of rows with an odd number of cells -/
theorem min_diagonal_pairs (t : CellTriangle) (h : t.rows = 5784) :
  oddRows t = 2892 := by sorry

end min_diagonal_pairs_l3621_362126


namespace multiply_24_to_get_2376_l3621_362189

theorem multiply_24_to_get_2376 (x : ℚ) : 24 * x = 2376 → x = 99 := by
  sorry

end multiply_24_to_get_2376_l3621_362189


namespace b_completion_time_l3621_362103

-- Define the work rates for A and B
def work_rate_A : ℚ := 1 / 2
def work_rate_B : ℚ := 1 / 6

-- Define the total work as 1
def total_work : ℚ := 1

-- Define the work done in one day by both A and B
def work_done_together : ℚ := work_rate_A + work_rate_B

-- Define the remaining work after one day
def remaining_work : ℚ := total_work - work_done_together

-- Theorem to prove
theorem b_completion_time :
  remaining_work / work_rate_B = 2 := by sorry

end b_completion_time_l3621_362103


namespace solution_set_of_increasing_function_l3621_362173

theorem solution_set_of_increasing_function 
  (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_f_0 : f 0 = -1) 
  (h_f_3 : f 3 = 1) : 
  {x : ℝ | |f (x + 1)| < 1} = Set.Ioo (-1) 2 := by
sorry

end solution_set_of_increasing_function_l3621_362173


namespace johns_candy_store_spending_l3621_362166

/-- Proves that John's candy store spending is $0.88 given his allowance and spending pattern -/
theorem johns_candy_store_spending (allowance : ℚ) : 
  allowance = 33/10 →
  let arcade_spending := 3/5 * allowance
  let remaining_after_arcade := allowance - arcade_spending
  let toy_store_spending := 1/3 * remaining_after_arcade
  let candy_store_spending := remaining_after_arcade - toy_store_spending
  candy_store_spending = 88/100 := by
  sorry

end johns_candy_store_spending_l3621_362166


namespace perpendicular_lines_sum_l3621_362180

/-- Given two perpendicular lines and the foot of the perpendicular, prove that a + b + c = -4 -/
theorem perpendicular_lines_sum (a b c : ℝ) : 
  (∀ x y, a * x + 4 * y - 2 = 0 ↔ 2 * x - 5 * y + b = 0) →  -- lines are perpendicular
  (a + 4 * c - 2 = 0) →  -- foot of perpendicular satisfies first line equation
  (2 - 5 * c + b = 0) →  -- foot of perpendicular satisfies second line equation
  (a * 2 + 4 * 5 = 0) →  -- perpendicularity condition
  a + b + c = -4 :=
by sorry

end perpendicular_lines_sum_l3621_362180


namespace rhombus_perimeter_l3621_362144

/-- A rhombus with diagonals in the ratio 3:4 and sum 56 has perimeter 80 -/
theorem rhombus_perimeter (d₁ d₂ s : ℝ) : 
  d₁ > 0 → d₂ > 0 → s > 0 →
  d₁ / d₂ = 3 / 4 → 
  d₁ + d₂ = 56 → 
  s^2 = (d₁/2)^2 + (d₂/2)^2 → 
  4 * s = 80 := by sorry

end rhombus_perimeter_l3621_362144


namespace opposite_of_negative_six_l3621_362141

theorem opposite_of_negative_six : -((-6) : ℤ) = 6 := by sorry

end opposite_of_negative_six_l3621_362141


namespace z_equation_l3621_362185

theorem z_equation (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y - 2*x*y ≠ 0) :
  (1/x + 1/y = 2 + 1/z) → z = (x*y)/(x + y - 2*x*y) := by
  sorry

end z_equation_l3621_362185


namespace sqrt_inequality_l3621_362124

theorem sqrt_inequality (a : ℝ) (h : a ≥ 2) :
  Real.sqrt (a - 2) - Real.sqrt a < Real.sqrt (a - 1) - Real.sqrt (a + 1) := by
  sorry

end sqrt_inequality_l3621_362124


namespace clock_angle_at_7_proof_l3621_362111

/-- The smaller angle formed by the hands of a clock at 7 o'clock -/
def clock_angle_at_7 : ℝ := 150

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℝ := 360

/-- The number of hour points on a clock -/
def clock_hour_points : ℕ := 12

/-- The position of the hour hand at 7 o'clock -/
def hour_hand_position : ℕ := 7

/-- The position of the minute hand at 7 o'clock -/
def minute_hand_position : ℕ := 12

theorem clock_angle_at_7_proof :
  clock_angle_at_7 = (minute_hand_position - hour_hand_position) * (full_circle_degrees / clock_hour_points) :=
by sorry

end clock_angle_at_7_proof_l3621_362111


namespace contest_prize_money_l3621_362104

/-- The total prize money for a novel contest -/
def total_prize_money (
  total_novels : ℕ
  ) (first_prize second_prize third_prize remaining_prize : ℕ
  ) : ℕ :=
  first_prize + second_prize + third_prize + (total_novels - 3) * remaining_prize

/-- Theorem stating that the total prize money for the given contest is $800 -/
theorem contest_prize_money :
  total_prize_money 18 200 150 120 22 = 800 := by
  sorry

end contest_prize_money_l3621_362104


namespace cos_angle_between_vectors_l3621_362193

theorem cos_angle_between_vectors (a b : ℝ × ℝ) :
  a = (-2, 1) →
  a + 2 • b = (2, 3) →
  let cos_theta := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  cos_theta = -3/5 := by
  sorry

end cos_angle_between_vectors_l3621_362193


namespace a0_value_l3621_362133

theorem a0_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^5 = a₀ + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a₀ = 32 := by
sorry

end a0_value_l3621_362133


namespace no_solution_exists_l3621_362117

theorem no_solution_exists : ¬∃ n : ℤ,
  50 ≤ n ∧ n ≤ 150 ∧
  8 ∣ n ∧
  n % 10 = 6 ∧
  n % 7 = 6 := by
  sorry

end no_solution_exists_l3621_362117


namespace john_old_cards_l3621_362161

/-- The number of baseball cards John puts on each page of the binder -/
def cards_per_page : ℕ := 3

/-- The number of new cards John has -/
def new_cards : ℕ := 8

/-- The total number of pages John used in the binder -/
def total_pages : ℕ := 8

/-- The number of old cards John had -/
def old_cards : ℕ := total_pages * cards_per_page - new_cards

theorem john_old_cards : old_cards = 16 := by
  sorry

end john_old_cards_l3621_362161


namespace abc_sum_sqrt_l3621_362140

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 15)
  (h2 : c + a = 18)
  (h3 : a + b = 21) :
  Real.sqrt (a * b * c * (a + b + c)) = 162 * Real.sqrt 2 := by
  sorry

end abc_sum_sqrt_l3621_362140


namespace repeating_decimal_equals_fraction_l3621_362187

/-- Represents a number with an integer part and a repeating decimal part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The given repeating decimal 5.341341341... -/
def givenNumber : RepeatingDecimal :=
  { integerPart := 5, repeatingPart := 341 }

/-- Theorem stating that 5.341341341... equals 5336/999 -/
theorem repeating_decimal_equals_fraction :
  toRational givenNumber = 5336 / 999 := by
  sorry

end repeating_decimal_equals_fraction_l3621_362187


namespace same_side_of_line_l3621_362164

/-- 
Given a line 2x - y + 1 = 0 and two points (1, 2) and (1, 0),
prove that these points are on the same side of the line.
-/
theorem same_side_of_line (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = 1 ∧ y₁ = 2 ∧ x₂ = 1 ∧ y₂ = 0 →
  (2 * x₁ - y₁ + 1 > 0) ∧ (2 * x₂ - y₂ + 1 > 0) :=
by sorry

end same_side_of_line_l3621_362164


namespace product_digit_sum_base7_l3621_362195

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a base-7 number --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base-7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := 
  toBase7 (toBase10 a * toBase10 b)

theorem product_digit_sum_base7 : 
  sumDigitsBase7 (multiplyBase7 35 42) = 21 := by sorry

end product_digit_sum_base7_l3621_362195


namespace watchtower_probability_l3621_362127

/-- Represents a searchlight with a given rotation speed in revolutions per minute -/
structure Searchlight where
  speed : ℝ
  speed_positive : speed > 0

/-- The setup of the watchtower problem -/
structure WatchtowerSetup where
  searchlight1 : Searchlight
  searchlight2 : Searchlight
  searchlight3 : Searchlight
  path_time : ℝ
  sl1_speed : searchlight1.speed = 2
  sl2_speed : searchlight2.speed = 3
  sl3_speed : searchlight3.speed = 4
  path_time_value : path_time = 30

/-- The probability of all searchlights not completing a revolution within the given time is 0 -/
theorem watchtower_probability (setup : WatchtowerSetup) :
  ∃ (s : Searchlight), s ∈ [setup.searchlight1, setup.searchlight2, setup.searchlight3] ∧
  (60 / s.speed ≤ setup.path_time) :=
sorry

end watchtower_probability_l3621_362127


namespace rice_division_l3621_362192

/-- 
Given an arithmetic sequence of three terms (a, b, c) where:
- The sum of the terms is 180
- The difference between the first and third term is 36
This theorem proves that the middle term (b) is equal to 60.
-/
theorem rice_division (a b c : ℕ) : 
  a + b + c = 180 →
  a - c = 36 →
  b = 60 := by
  sorry


end rice_division_l3621_362192


namespace no_k_exists_for_not_in_second_quadrant_l3621_362105

/-- A linear function that does not pass through the second quadrant -/
def not_in_second_quadrant (k : ℝ) : Prop :=
  ∀ x y : ℝ, y = (k - 1) * x + k → (x < 0 → y ≤ 0)

/-- Theorem stating that there is no k for which the linear function y=(k-1)x+k does not pass through the second quadrant -/
theorem no_k_exists_for_not_in_second_quadrant :
  ¬ ∃ k : ℝ, not_in_second_quadrant k :=
sorry

end no_k_exists_for_not_in_second_quadrant_l3621_362105


namespace locus_is_ellipse_l3621_362191

noncomputable section

/-- Two circles with common center and a point configuration --/
structure CircleConfig where
  a : ℝ
  b : ℝ
  h_ab : a > b

variable (cfg : CircleConfig)

/-- The locus of points Si --/
def locus (cfg : CircleConfig) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ cfg.b^2 / cfg.a ∧
    p.1 = t * cfg.a^2 / cfg.b^2 ∧
    p.2^2 = cfg.b^2 - (t * cfg.a / cfg.b)^2}

/-- The ellipse with major axis 2a and minor axis 2b --/
def ellipse (cfg : CircleConfig) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / cfg.a^2 + p.2^2 / cfg.b^2 = 1 ∧ p.1 ≥ 0}

/-- The main theorem --/
theorem locus_is_ellipse (cfg : CircleConfig) :
  locus cfg = ellipse cfg := by sorry

end

end locus_is_ellipse_l3621_362191


namespace new_average_after_changes_l3621_362156

theorem new_average_after_changes (numbers : Finset ℕ) (original_sum : ℕ) : 
  numbers.card = 15 → 
  original_sum = numbers.sum id →
  original_sum / numbers.card = 40 →
  let new_sum := original_sum + 9 * 10 - 6 * 5
  new_sum / numbers.card = 44 := by
sorry

end new_average_after_changes_l3621_362156


namespace cube_structure_surface_area_total_surface_area_is_1266_l3621_362147

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the volume of a cube -/
def Cube.volume (c : Cube) : ℕ := c.sideLength ^ 3

/-- Calculates the surface area of a cube -/
def Cube.surfaceArea (c : Cube) : ℕ := 6 * c.sideLength ^ 2

/-- Represents the structure formed by the cubes -/
structure CubeStructure where
  cubes : List Cube
  stackedCubes : List Cube
  adjacentCube : Cube
  topCube : Cube

/-- Theorem stating the total surface area of the cube structure -/
theorem cube_structure_surface_area (cs : CubeStructure) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem total_surface_area_is_1266 (cs : CubeStructure) 
  (h1 : cs.cubes.length = 8)
  (h2 : (cs.cubes.map Cube.volume) = [1, 8, 27, 64, 125, 216, 512, 729])
  (h3 : cs.stackedCubes.length = 6)
  (h4 : cs.stackedCubes = (cs.cubes.take 6).reverse)
  (h5 : cs.adjacentCube = cs.cubes[6])
  (h6 : cs.adjacentCube.sideLength = 6)
  (h7 : cs.stackedCubes[4].sideLength = 5)
  (h8 : cs.topCube = cs.cubes[7])
  (h9 : cs.topCube.sideLength = 8) :
  cube_structure_surface_area cs = 1266 :=
sorry

end cube_structure_surface_area_total_surface_area_is_1266_l3621_362147


namespace complex_absolute_value_l3621_362120

/-- Given a complex number z such that (1 + 2i) / z = 1 + i,
    prove that the absolute value of z is equal to √10 / 2. -/
theorem complex_absolute_value (z : ℂ) (h : (1 + 2 * Complex.I) / z = 1 + Complex.I) :
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_absolute_value_l3621_362120


namespace M_intersect_N_eq_M_l3621_362155

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def N : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- Theorem statement
theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end M_intersect_N_eq_M_l3621_362155


namespace tenth_group_draw_l3621_362153

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : Nat
  sample_size : Nat
  group_size : Nat
  first_draw : Nat

/-- Calculates the number drawn from a specific group in systematic sampling -/
def draw_from_group (s : SystematicSampling) (group : Nat) : Nat :=
  s.first_draw + s.group_size * (group - 1)

theorem tenth_group_draw (s : SystematicSampling) 
  (h1 : s.total_students = 1000)
  (h2 : s.sample_size = 100)
  (h3 : s.group_size = 10)
  (h4 : s.first_draw = 6) :
  draw_from_group s 10 = 96 := by
  sorry

end tenth_group_draw_l3621_362153


namespace tangent_line_equation_l3621_362136

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2 * x

/-- The point through which the line passes -/
def point : ℝ × ℝ := (1, 1)

/-- The equation of the line: 2x - y - 1 = 0 -/
def line_equation (x y : ℝ) : Prop := 2 * x - y - 1 = 0

theorem tangent_line_equation :
  (∀ x y, line_equation x y ↔ 
    (y - point.2 = f' point.1 * (x - point.1) ∧
     f point.1 = point.2)) :=
by sorry

end tangent_line_equation_l3621_362136


namespace greatest_multiple_of_three_cubed_less_than_1000_l3621_362165

theorem greatest_multiple_of_three_cubed_less_than_1000 :
  ∃ (x : ℕ), 
    x > 0 ∧ 
    ∃ (k : ℕ), x = 3 * k ∧ 
    x^3 < 1000 ∧
    ∀ (y : ℕ), y > 0 → (∃ (m : ℕ), y = 3 * m) → y^3 < 1000 → y ≤ x ∧
    x = 9 :=
by sorry

end greatest_multiple_of_three_cubed_less_than_1000_l3621_362165


namespace smallest_perimeter_square_sides_l3621_362139

theorem smallest_perimeter_square_sides : ∃ (a b c : ℕ), 
  (0 < a ∧ 0 < b ∧ 0 < c) ∧  -- positive integers
  (a < b ∧ b < c) ∧  -- distinct
  (a^2 + b^2 > c^2) ∧  -- triangle inequality
  (a^2 + c^2 > b^2) ∧
  (b^2 + c^2 > a^2) ∧
  (a^2 + b^2 + c^2 = 77) ∧  -- perimeter is 77
  (∀ (x y z : ℕ), (0 < x ∧ 0 < y ∧ 0 < z) →
    (x < y ∧ y < z) →
    (x^2 + y^2 > z^2) →
    (x^2 + z^2 > y^2) →
    (y^2 + z^2 > x^2) →
    (x^2 + y^2 + z^2 ≥ 77)) :=
by
  sorry

#check smallest_perimeter_square_sides

end smallest_perimeter_square_sides_l3621_362139
