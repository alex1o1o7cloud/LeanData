import Mathlib

namespace NUMINAMATH_CALUDE_andy_profit_per_cake_l2686_268620

/-- Andy's cake business model -/
structure CakeBusiness where
  ingredient_cost_two_cakes : ℕ
  packaging_cost_per_cake : ℕ
  selling_price_per_cake : ℕ

/-- Calculate the profit per cake -/
def profit_per_cake (b : CakeBusiness) : ℕ :=
  b.selling_price_per_cake - (b.ingredient_cost_two_cakes / 2 + b.packaging_cost_per_cake)

/-- Theorem: Andy's profit per cake is $8 -/
theorem andy_profit_per_cake :
  ∃ (b : CakeBusiness),
    b.ingredient_cost_two_cakes = 12 ∧
    b.packaging_cost_per_cake = 1 ∧
    b.selling_price_per_cake = 15 ∧
    profit_per_cake b = 8 := by
  sorry

end NUMINAMATH_CALUDE_andy_profit_per_cake_l2686_268620


namespace NUMINAMATH_CALUDE_ashok_subjects_l2686_268640

theorem ashok_subjects (total_average : ℝ) (five_subjects_average : ℝ) (sixth_subject_mark : ℝ) 
  (h1 : total_average = 70)
  (h2 : five_subjects_average = 74)
  (h3 : sixth_subject_mark = 50) :
  ∃ (n : ℕ), n = 6 ∧ n * total_average = 5 * five_subjects_average + sixth_subject_mark :=
by
  sorry

end NUMINAMATH_CALUDE_ashok_subjects_l2686_268640


namespace NUMINAMATH_CALUDE_minimum_days_to_exceed_500_l2686_268675

def bacteria_count (initial_count : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial_count * growth_factor ^ days

theorem minimum_days_to_exceed_500 :
  ∃ (n : ℕ), n = 6 ∧
  (∀ (k : ℕ), k < n → bacteria_count 4 3 k ≤ 500) ∧
  bacteria_count 4 3 n > 500 :=
sorry

end NUMINAMATH_CALUDE_minimum_days_to_exceed_500_l2686_268675


namespace NUMINAMATH_CALUDE_box_plates_cups_weight_l2686_268649

/-- Given the weights of various combinations of a box, plates, and cups, 
    prove that the weight of the box with 10 plates and 20 cups is 3 kg. -/
theorem box_plates_cups_weight :
  ∀ (b p c : ℝ),
  (b + 20 * p + 30 * c = 4.8) →
  (b + 40 * p + 50 * c = 8.4) →
  (b + 10 * p + 20 * c = 3) :=
by sorry

end NUMINAMATH_CALUDE_box_plates_cups_weight_l2686_268649


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l2686_268661

theorem sum_of_two_numbers (a b : ℕ) : a = 30 ∧ b = 42 ∧ b = a + 12 → a + b = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l2686_268661


namespace NUMINAMATH_CALUDE_min_intercept_line_l2686_268638

/-- A line that passes through a point and intersects the positive halves of the coordinate axes -/
structure InterceptLine where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : 1 / a + 9 / b = 1

/-- The sum of intercepts of an InterceptLine -/
def sum_of_intercepts (l : InterceptLine) : ℝ := l.a + l.b

/-- The equation of the line with minimum sum of intercepts -/
def min_intercept_line_eq (x y : ℝ) : Prop := 3 * x + y - 12 = 0

theorem min_intercept_line :
  ∃ (l : InterceptLine), ∀ (l' : InterceptLine), 
    sum_of_intercepts l ≤ sum_of_intercepts l' ∧
    min_intercept_line_eq l.a l.b := by sorry

end NUMINAMATH_CALUDE_min_intercept_line_l2686_268638


namespace NUMINAMATH_CALUDE_rank_from_bottom_calculation_l2686_268603

/-- Represents a student's ranking in a class. -/
structure StudentRanking where
  totalStudents : Nat
  rankFromTop : Nat
  rankFromBottom : Nat

/-- Calculates the rank from the bottom given the total number of students and rank from the top. -/
def calculateRankFromBottom (total : Nat) (rankFromTop : Nat) : Nat :=
  total - rankFromTop + 1

/-- Theorem stating that for a class of 53 students, a student ranking 5th from the top
    will rank 49th from the bottom. -/
theorem rank_from_bottom_calculation (s : StudentRanking)
    (h1 : s.totalStudents = 53)
    (h2 : s.rankFromTop = 5)
    (h3 : s.rankFromBottom = calculateRankFromBottom s.totalStudents s.rankFromTop) :
  s.rankFromBottom = 49 := by
  sorry

#check rank_from_bottom_calculation

end NUMINAMATH_CALUDE_rank_from_bottom_calculation_l2686_268603


namespace NUMINAMATH_CALUDE_root_in_interval_l2686_268688

def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem root_in_interval (k : ℕ) : 
  (∃ x : ℝ, x > k ∧ x < k + 1 ∧ f x = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2686_268688


namespace NUMINAMATH_CALUDE_caspers_candies_l2686_268651

theorem caspers_candies (initial_candies : ℕ) : 
  let day1_after_eating := (3 * initial_candies) / 4
  let day1_remaining := day1_after_eating - 3
  let day2_after_eating := (4 * day1_remaining) / 5
  let day2_remaining := day2_after_eating - 5
  let day3_after_giving := day2_remaining - 7
  let final_candies := (5 * day3_after_giving) / 6
  final_candies = 10 → initial_candies = 44 := by
sorry

end NUMINAMATH_CALUDE_caspers_candies_l2686_268651


namespace NUMINAMATH_CALUDE_billion_scientific_notation_l2686_268666

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_scientific_notation :
  toScientificNotation 1075000000 = ScientificNotation.mk 1.075 9 sorry sorry := by
  sorry

end NUMINAMATH_CALUDE_billion_scientific_notation_l2686_268666


namespace NUMINAMATH_CALUDE_championship_winner_l2686_268672

-- Define the teams
inductive Team : Type
| A | B | C | D

-- Define the positions
inductive Position : Type
| First | Second | Third | Fourth

-- Define a prediction as a pair of (Team, Position)
def Prediction := Team × Position

-- Define the predictions made by each person
def WangPredictions : Prediction × Prediction := ((Team.D, Position.First), (Team.B, Position.Second))
def LiPredictions : Prediction × Prediction := ((Team.A, Position.Second), (Team.C, Position.Fourth))
def ZhangPredictions : Prediction × Prediction := ((Team.C, Position.Third), (Team.D, Position.Second))

-- Define a function to check if a prediction is correct
def isPredictionCorrect (prediction : Prediction) (result : Team → Position) : Prop :=
  result prediction.1 = prediction.2

-- Define the theorem
theorem championship_winner (result : Team → Position) : 
  (isPredictionCorrect WangPredictions.1 result ≠ isPredictionCorrect WangPredictions.2 result) ∧
  (isPredictionCorrect LiPredictions.1 result ≠ isPredictionCorrect LiPredictions.2 result) ∧
  (isPredictionCorrect ZhangPredictions.1 result ≠ isPredictionCorrect ZhangPredictions.2 result) →
  result Team.D = Position.First :=
by
  sorry

end NUMINAMATH_CALUDE_championship_winner_l2686_268672


namespace NUMINAMATH_CALUDE_expression_evaluation_l2686_268625

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 = 1 / y^2) :
  (x^2 - 4/x^2) * (y^2 + 4/y^2) = x^4 - 16/x^4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2686_268625


namespace NUMINAMATH_CALUDE_weekend_study_hours_per_day_l2686_268613

-- Define the given conditions
def weekday_study_hours_per_night : ℕ := 2
def weekday_study_nights_per_week : ℕ := 5
def weeks_until_exam : ℕ := 6
def total_study_hours : ℕ := 96

-- Define the number of days in a weekend
def days_per_weekend : ℕ := 2

-- Define the theorem
theorem weekend_study_hours_per_day :
  (total_study_hours - weekday_study_hours_per_night * weekday_study_nights_per_week * weeks_until_exam) / (days_per_weekend * weeks_until_exam) = 3 := by
  sorry

end NUMINAMATH_CALUDE_weekend_study_hours_per_day_l2686_268613


namespace NUMINAMATH_CALUDE_race_speed_ratio_l2686_268677

/-- Proves the speed ratio in a race with given conditions -/
theorem race_speed_ratio (L : ℝ) (h : L > 0) : 
  ∃ R : ℝ, 
    (R > 0) ∧ 
    (0.26 * L = (1 - 0.74) * L) ∧
    (R * L = (1 - 0.60) * L) →
    R = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l2686_268677


namespace NUMINAMATH_CALUDE_square_of_99_9_l2686_268642

theorem square_of_99_9 : (99.9 : ℝ)^2 = 10000 - 20 + 0.01 := by
  sorry

end NUMINAMATH_CALUDE_square_of_99_9_l2686_268642


namespace NUMINAMATH_CALUDE_travel_ways_proof_l2686_268630

/-- The number of roads from village A to village B -/
def roads_A_to_B : ℕ := 3

/-- The number of roads from village B to village C -/
def roads_B_to_C : ℕ := 2

/-- The total number of ways to travel from village A to village C via village B -/
def total_ways : ℕ := roads_A_to_B * roads_B_to_C

theorem travel_ways_proof : total_ways = 6 := by
  sorry

end NUMINAMATH_CALUDE_travel_ways_proof_l2686_268630


namespace NUMINAMATH_CALUDE_arrangement_counts_l2686_268607

/-- The number of ways to arrange 3 boys and 4 girls in a row under specific conditions -/
theorem arrangement_counts :
  let total_people : ℕ := 7
  let num_boys : ℕ := 3
  let num_girls : ℕ := 4
  -- (1) Person A is neither at the middle nor at the ends
  (number_of_arrangements_1 : ℕ := 2880) →
  -- (2) Persons A and B must be at the two ends
  (number_of_arrangements_2 : ℕ := 240) →
  -- (3) Boys and girls alternate
  (number_of_arrangements_3 : ℕ := 144) →
  -- Prove all three conditions are true
  (number_of_arrangements_1 = 2880 ∧
   number_of_arrangements_2 = 240 ∧
   number_of_arrangements_3 = 144) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_counts_l2686_268607


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l2686_268647

theorem smallest_four_digit_multiple_of_18 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 18 ∣ n → n ≥ 1008 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l2686_268647


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2686_268627

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) :
  π * r^2 = 256 * π → 2 * π * r^2 + π * r^2 = 768 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2686_268627


namespace NUMINAMATH_CALUDE_percentage_difference_l2686_268600

theorem percentage_difference (x y : ℝ) (h : y = 1.8 * x) : 
  (x - y) / y * 100 = -(4 / 9) * 100 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l2686_268600


namespace NUMINAMATH_CALUDE_smallest_divisible_by_72_l2686_268655

/-- Concatenates the digits of natural numbers from 1 to n -/
def concatenateDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is divisible by 72 -/
def isDivisibleBy72 (n : ℕ) : Prop := n % 72 = 0

/-- Theorem stating that 36 is the smallest positive integer n such that 
    the concatenated digits from 1 to n form a number divisible by 72 -/
theorem smallest_divisible_by_72 :
  ∀ k < 36, ¬ isDivisibleBy72 (concatenateDigits k) ∧
  isDivisibleBy72 (concatenateDigits 36) := by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_72_l2686_268655


namespace NUMINAMATH_CALUDE_polynomial_equality_l2686_268612

theorem polynomial_equality (a b c d : ℝ) : 
  (∀ x : ℝ, x^4 + 4*x^3 + 3*x^2 + 2*x + 1 = 
    (x+1)^4 + a*(x+1)^3 + b*(x+1)^2 + c*(x+1) + d) → 
  (a = 0 ∧ b = -3 ∧ c = 4 ∧ d = -1) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2686_268612


namespace NUMINAMATH_CALUDE_neighborhood_total_l2686_268636

/-- Represents the number of households in different categories -/
structure Neighborhood where
  neither : ℕ
  both : ℕ
  with_car : ℕ
  bike_only : ℕ

/-- Calculates the total number of households in the neighborhood -/
def total_households (n : Neighborhood) : ℕ :=
  n.neither + (n.with_car - n.both) + n.bike_only + n.both

/-- Theorem stating that the total number of households is 90 -/
theorem neighborhood_total (n : Neighborhood) 
  (h1 : n.neither = 11)
  (h2 : n.both = 14)
  (h3 : n.with_car = 44)
  (h4 : n.bike_only = 35) : 
  total_households n = 90 := by
  sorry

#eval total_households { neither := 11, both := 14, with_car := 44, bike_only := 35 }

end NUMINAMATH_CALUDE_neighborhood_total_l2686_268636


namespace NUMINAMATH_CALUDE_solution_set_when_m_2_solution_set_condition_l2686_268628

-- Define the function f
def f (x m : ℝ) : ℝ := |2*x - m| + 4*x

-- Part I
theorem solution_set_when_m_2 :
  {x : ℝ | f x 2 ≤ 1} = {x : ℝ | x ≤ -1/2} := by sorry

-- Part II
theorem solution_set_condition (m : ℝ) :
  {x : ℝ | f x m ≤ 2} = {x : ℝ | x ≤ -2} ↔ m = 6 ∨ m = -14 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_2_solution_set_condition_l2686_268628


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2686_268653

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a*b/c + b*c/a + c*a/b = 49/6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2686_268653


namespace NUMINAMATH_CALUDE_multiplication_problem_l2686_268662

theorem multiplication_problem : ∃ x : ℕ, 987 * x = 555681 ∧ x = 563 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l2686_268662


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_range_l2686_268615

/-- Given an infinite geometric sequence {a_n} with common ratio q,
    if the sum of all terms is equal to q, then the range of the first term a_1 is:
    -2 < a_1 ≤ 1/4 and a_1 ≠ 0 -/
theorem geometric_sequence_first_term_range (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- Common ratio is q
  (∃ S : ℝ, S = q ∧ S = ∑' n, a n) →  -- Sum of all terms is q
  (-2 < a 0 ∧ a 0 ≤ 1/4 ∧ a 0 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_range_l2686_268615


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l2686_268654

theorem quadratic_equation_condition (m : ℝ) : 
  (abs m + 2 = 2 ∧ m - 3 ≠ 0) ↔ m = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l2686_268654


namespace NUMINAMATH_CALUDE_set_operations_l2686_268604

def A : Set ℝ := {x | x > 4}
def B : Set ℝ := {x | -6 < x ∧ x < 6}

theorem set_operations :
  (A ∩ B = {x | 4 < x ∧ x < 6}) ∧
  (Set.univ \ B = {x | x ≥ 6 ∨ x ≤ -6}) ∧
  (A \ B = {x | x ≥ 6}) ∧
  (A \ (A \ B) = {x | 4 < x ∧ x < 6}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l2686_268604


namespace NUMINAMATH_CALUDE_unique_isolating_line_condition_l2686_268644

/-- The isolating line property for two functions -/
def IsIsolatingLine (f g : ℝ → ℝ) (k b : ℝ) : Prop :=
  ∀ x, f x ≥ k * x + b ∧ k * x + b ≥ g x

/-- The existence of a unique isolating line for x^2 and a * ln(x) -/
theorem unique_isolating_line_condition (a : ℝ) :
  (a > 0) →
  (∃! k b, IsIsolatingLine (fun x ↦ x^2) (fun x ↦ a * Real.log x) k b) ↔
  a = 2 * Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_isolating_line_condition_l2686_268644


namespace NUMINAMATH_CALUDE_tallest_giraffe_height_is_96_l2686_268671

/-- The height of the shortest giraffe in inches -/
def shortest_giraffe_height : ℕ := 68

/-- The height difference between the tallest and shortest giraffes in inches -/
def height_difference : ℕ := 28

/-- The number of adult giraffes at the zoo -/
def num_giraffes : ℕ := 14

/-- The height of the tallest giraffe in inches -/
def tallest_giraffe_height : ℕ := shortest_giraffe_height + height_difference

theorem tallest_giraffe_height_is_96 : tallest_giraffe_height = 96 := by
  sorry

end NUMINAMATH_CALUDE_tallest_giraffe_height_is_96_l2686_268671


namespace NUMINAMATH_CALUDE_vector_complex_correspondence_l2686_268681

theorem vector_complex_correspondence (z : ℂ) :
  z = -3 + 2*I → (-z) = 3 - 2*I := by sorry

end NUMINAMATH_CALUDE_vector_complex_correspondence_l2686_268681


namespace NUMINAMATH_CALUDE_shortest_segment_length_l2686_268684

/-- Represents the paper strip and folding operations -/
structure PaperStrip where
  length : Real
  red_dot_position : Real
  yellow_dot_position : Real

/-- Calculates the position of the yellow dot after the first fold -/
def calculate_yellow_dot_position (strip : PaperStrip) : Real :=
  strip.length - strip.red_dot_position

/-- Calculates the length of the segment between red and yellow dots -/
def calculate_middle_segment (strip : PaperStrip) : Real :=
  strip.length - 2 * strip.yellow_dot_position

/-- Calculates the length of the shortest segment after all folds and cuts -/
def calculate_shortest_segment (strip : PaperStrip) : Real :=
  strip.red_dot_position - 2 * (strip.red_dot_position - strip.yellow_dot_position)

/-- Theorem stating that the shortest segment is 0.146 meters long -/
theorem shortest_segment_length :
  let initial_strip : PaperStrip := {
    length := 1,
    red_dot_position := 0.618,
    yellow_dot_position := calculate_yellow_dot_position { length := 1, red_dot_position := 0.618, yellow_dot_position := 0 }
  }
  calculate_shortest_segment initial_strip = 0.146 := by
  sorry

end NUMINAMATH_CALUDE_shortest_segment_length_l2686_268684


namespace NUMINAMATH_CALUDE_smallest_k_for_two_roots_l2686_268616

/-- A quadratic trinomial with natural number coefficients -/
structure QuadraticTrinomial where
  k : ℕ
  p : ℕ
  q : ℕ

/-- Predicate to check if a quadratic trinomial has two distinct positive roots less than 1 -/
def has_two_distinct_positive_roots_less_than_one (qt : QuadraticTrinomial) : Prop :=
  ∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧
    qt.k * x₁^2 - qt.p * x₁ + qt.q = 0 ∧
    qt.k * x₂^2 - qt.p * x₂ + qt.q = 0

/-- The main theorem stating that 5 is the smallest natural number k satisfying the condition -/
theorem smallest_k_for_two_roots : 
  (∀ k < 5, ¬∃ (p q : ℕ), has_two_distinct_positive_roots_less_than_one ⟨k, p, q⟩) ∧
  (∃ (p q : ℕ), has_two_distinct_positive_roots_less_than_one ⟨5, p, q⟩) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_two_roots_l2686_268616


namespace NUMINAMATH_CALUDE_triangle_area_l2686_268665

/-- The area of a triangle with vertices at (2, 1), (2, 7), and (8, 4) is 18 square units -/
theorem triangle_area : ℝ := by
  -- Define the vertices of the triangle
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (2, 7)
  let C : ℝ × ℝ := (8, 4)

  -- Calculate the area using the formula: Area = (1/2) * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
  let area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

  -- Prove that the calculated area equals 18
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2686_268665


namespace NUMINAMATH_CALUDE_A_intersect_B_l2686_268606

def A : Set ℕ := {x | x - 4 < 0}
def B : Set ℕ := {0, 1, 3, 4}

theorem A_intersect_B : A ∩ B = {0, 1, 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2686_268606


namespace NUMINAMATH_CALUDE_vector_at_t_4_l2686_268683

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  point : ℝ → (ℝ × ℝ × ℝ)

/-- The given line satisfying the conditions -/
def given_line : ParametricLine :=
  { point := sorry }

theorem vector_at_t_4 :
  given_line.point 1 = (4, 5, 9) →
  given_line.point 3 = (1, 0, -2) →
  given_line.point 4 = (-1, 0, -15) :=
by sorry

end NUMINAMATH_CALUDE_vector_at_t_4_l2686_268683


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l2686_268601

-- Define the sets A and B
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^(1/3)}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}

-- Define the open interval (1, +∞)
def open_interval : Set ℝ := {x : ℝ | x > 1}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = open_interval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l2686_268601


namespace NUMINAMATH_CALUDE_modulus_of_z_l2686_268658

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2686_268658


namespace NUMINAMATH_CALUDE_circle_through_points_l2686_268624

/-- A circle passing through three points -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Check if a point lies on the circle -/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  x^2 + y^2 + c.D * x + c.E * y + c.F = 0

/-- The specific circle we're interested in -/
def our_circle : Circle := { D := -4, E := -6, F := 0 }

theorem circle_through_points : 
  (our_circle.contains 0 0) ∧ 
  (our_circle.contains 4 0) ∧ 
  (our_circle.contains (-1) 1) :=
by sorry

#check circle_through_points

end NUMINAMATH_CALUDE_circle_through_points_l2686_268624


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2686_268639

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 - x| ≥ 1} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2686_268639


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l2686_268690

theorem marble_fraction_after_tripling (total : ℚ) (h1 : total > 0) : 
  let blue := (4/7) * total
  let green := total - blue
  let new_green := 3 * green
  let new_total := blue + new_green
  new_green / new_total = 9/13 := by
sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l2686_268690


namespace NUMINAMATH_CALUDE_smallest_multiple_thirty_six_satisfies_thirty_six_is_smallest_l2686_268694

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 450 * x % 648 = 0 → x ≥ 36 := by
  sorry

theorem thirty_six_satisfies : 450 * 36 % 648 = 0 := by
  sorry

theorem thirty_six_is_smallest : ∃ (x : ℕ), x > 0 ∧ 450 * x % 648 = 0 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_thirty_six_satisfies_thirty_six_is_smallest_l2686_268694


namespace NUMINAMATH_CALUDE_distance_sum_inequality_l2686_268641

theorem distance_sum_inequality (a : ℝ) (ha : a > 0) :
  (∃ x : ℝ, |x - 5| + |x - 1| < a) ↔ a > 4 := by sorry

end NUMINAMATH_CALUDE_distance_sum_inequality_l2686_268641


namespace NUMINAMATH_CALUDE_min_value_problem_l2686_268614

theorem min_value_problem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 36) : 
  (a*e)^2 + (b*f)^2 + (c*g)^2 + (d*h)^2 ≥ 576 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2686_268614


namespace NUMINAMATH_CALUDE_sector_area_specific_sector_area_l2686_268626

/-- Given a sector with central angle α and arc length l, 
    proves that the area of the sector is (l * l) / (2 * α) -/
theorem sector_area (α : ℝ) (l : ℝ) (h1 : α > 0) (h2 : l > 0) :
  let R := l / α
  (1 / 2) * l * R = (l * l) / (2 * α) := by sorry

/-- Proves that a sector with central angle 2 radians and arc length 4cm 
    has an area of 4cm² -/
theorem specific_sector_area :
  let α : ℝ := 2
  let l : ℝ := 4
  (1 / 2) * l * (l / α) = 4 := by sorry

end NUMINAMATH_CALUDE_sector_area_specific_sector_area_l2686_268626


namespace NUMINAMATH_CALUDE_delegate_seating_probability_l2686_268695

-- Define the number of delegates and countries
def total_delegates : ℕ := 12
def num_countries : ℕ := 3
def delegates_per_country : ℕ := 4

-- Define the probability as a fraction
def probability : ℚ := 106 / 115

-- State the theorem
theorem delegate_seating_probability :
  let total_arrangements := (total_delegates.factorial) / ((delegates_per_country.factorial) ^ num_countries)
  let unwanted_arrangements := 
    (num_countries * total_delegates * ((total_delegates - delegates_per_country).factorial / 
    (delegates_per_country.factorial ^ (num_countries - 1)))) -
    (num_countries * total_delegates * (delegates_per_country + 2)) +
    (total_delegates * 2)
  (total_arrangements - unwanted_arrangements) / total_arrangements = probability := by
  sorry

end NUMINAMATH_CALUDE_delegate_seating_probability_l2686_268695


namespace NUMINAMATH_CALUDE_point_outside_intersecting_line_l2686_268618

/-- A line ax + by = 1 intersects a unit circle if and only if the distance
    from the origin to the line is less than 1 -/
def line_intersects_circle (a b : ℝ) : Prop :=
  (|1| / Real.sqrt (a^2 + b^2)) < 1

/-- A point (x,y) is outside the unit circle if its distance from the origin is greater than 1 -/
def point_outside_circle (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) > 1

theorem point_outside_intersecting_line (a b : ℝ) :
  line_intersects_circle a b → point_outside_circle a b :=
by sorry

end NUMINAMATH_CALUDE_point_outside_intersecting_line_l2686_268618


namespace NUMINAMATH_CALUDE_solve_equation_l2686_268608

theorem solve_equation (x : ℝ) (h : 9 / (1 + 4 / x) = 1) : x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2686_268608


namespace NUMINAMATH_CALUDE_ravi_coin_value_l2686_268629

/-- Represents the number of coins of each type Ravi has -/
structure CoinCounts where
  nickels : ℕ
  quarters : ℕ
  dimes : ℕ
  half_dollars : ℕ
  pennies : ℕ

/-- Calculates the total value of coins in cents -/
def total_value (counts : CoinCounts) : ℕ :=
  counts.nickels * 5 +
  counts.quarters * 25 +
  counts.dimes * 10 +
  counts.half_dollars * 50 +
  counts.pennies * 1

/-- Theorem stating that Ravi's coin collection is worth $12.51 -/
theorem ravi_coin_value : ∃ (counts : CoinCounts),
  counts.nickels = 6 ∧
  counts.quarters = counts.nickels + 2 ∧
  counts.dimes = counts.quarters + 4 ∧
  counts.half_dollars = counts.dimes + 5 ∧
  counts.pennies = counts.half_dollars * 3 ∧
  total_value counts = 1251 := by
  sorry

end NUMINAMATH_CALUDE_ravi_coin_value_l2686_268629


namespace NUMINAMATH_CALUDE_equation_solution_l2686_268698

theorem equation_solution (x : ℝ) : 
  (3 / (x^2 + x) - x^2 = 2 + x) → (2*x^2 + 2*x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2686_268698


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l2686_268610

def M (p : ℝ) := {x : ℝ | x^2 - p*x + 6 = 0}
def N (q : ℝ) := {x : ℝ | x^2 + 6*x - q = 0}

theorem intersection_implies_sum (p q : ℝ) :
  M p ∩ N q = {2} → p + q = 21 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l2686_268610


namespace NUMINAMATH_CALUDE_triangle_side_length_l2686_268691

theorem triangle_side_length (a b c : ℝ) (A : Real) :
  a = Real.sqrt 5 →
  b = Real.sqrt 15 →
  A = 30 * Real.pi / 180 →
  c = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2686_268691


namespace NUMINAMATH_CALUDE_prob_same_color_is_89_169_l2686_268605

def blue_balls : ℕ := 8
def yellow_balls : ℕ := 5
def total_balls : ℕ := blue_balls + yellow_balls

def prob_same_color : ℚ := (blue_balls^2 + yellow_balls^2) / total_balls^2

theorem prob_same_color_is_89_169 : prob_same_color = 89 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_89_169_l2686_268605


namespace NUMINAMATH_CALUDE_cubic_fraction_sum_l2686_268668

theorem cubic_fraction_sum (a b : ℝ) (h1 : |a| ≠ |b|) 
  (h2 : (a + b) / (a - b) + (a - b) / (a + b) = 6) :
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 18 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_sum_l2686_268668


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_b_l2686_268645

def b (n : ℕ) : ℕ := n.factorial + 3 * n

theorem max_gcd_consecutive_b : (∃ n : ℕ, Nat.gcd (b n) (b (n + 1)) = 14) ∧ 
  (∀ n : ℕ, Nat.gcd (b n) (b (n + 1)) ≤ 14) :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_b_l2686_268645


namespace NUMINAMATH_CALUDE_right_triangle_angle_calculation_l2686_268673

theorem right_triangle_angle_calculation (α β γ : ℝ) :
  α = 90 ∧ β = 63 ∧ α + β + γ = 180 → γ = 27 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_calculation_l2686_268673


namespace NUMINAMATH_CALUDE_max_value_sum_l2686_268686

theorem max_value_sum (a b c d e : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e)
  (sum_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 504) : 
  ∃ (N a_N b_N c_N d_N e_N : ℝ),
    (∀ (x y z w v : ℝ), x * z + 3 * y * z + 4 * z * w + 8 * z * v ≤ N) ∧
    (N = a_N * c_N + 3 * b_N * c_N + 4 * c_N * d_N + 8 * c_N * e_N) ∧
    (a_N^2 + b_N^2 + c_N^2 + d_N^2 + e_N^2 = 504) ∧
    (N + a_N + b_N + c_N + d_N + e_N = 32 + 1512 * Real.sqrt 10 + 6 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_l2686_268686


namespace NUMINAMATH_CALUDE_fly_probabilities_l2686_268664

def fly_move (x y : ℕ) : Prop := x ≤ 8 ∧ y ≤ 10

def prob_reach (x y : ℕ) : ℚ := (Nat.choose (x + y) x : ℚ) / 2^(x + y)

def prob_through (x1 y1 x2 y2 x3 y3 : ℕ) : ℚ :=
  (Nat.choose (x1 + y1) x1 * Nat.choose (x3 - x2 + y3 - y2) (x3 - x2) : ℚ) / 2^(x3 + y3)

def inside_circle (x y cx cy r : ℝ) : Prop :=
  (x - cx)^2 + (y - cy)^2 ≤ r^2

theorem fly_probabilities :
  let p1 := prob_reach 8 10
  let p2 := prob_through 5 6 6 6 8 10
  let p3 := (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + Nat.choose 9 4 ^ 2 : ℚ) / 2^18
  (p1 = (Nat.choose 18 8 : ℚ) / 2^18) ∧
  (p2 = (Nat.choose 11 5 * Nat.choose 6 2 : ℚ) / 2^18) ∧
  (∀ x y, fly_move x y → inside_circle x y 4 5 3 → prob_reach x y ≤ p3) := by
  sorry

end NUMINAMATH_CALUDE_fly_probabilities_l2686_268664


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2686_268611

def num_arabic : ℕ := 2
def num_german : ℕ := 3
def num_spanish : ℕ := 4
def total_books : ℕ := num_arabic + num_german + num_spanish

def arrangement_count : ℕ := sorry

theorem book_arrangement_count :
  arrangement_count = 3456 := by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2686_268611


namespace NUMINAMATH_CALUDE_total_accidents_in_four_minutes_l2686_268663

/-- Represents the number of seconds in 4 minutes -/
def total_seconds : ℕ := 4 * 60

/-- Represents the frequency of car collisions in seconds -/
def car_collision_frequency : ℕ := 3

/-- Represents the frequency of big crashes in seconds -/
def big_crash_frequency : ℕ := 7

/-- Represents the frequency of multi-vehicle pile-ups in seconds -/
def pile_up_frequency : ℕ := 15

/-- Represents the frequency of massive accidents in seconds -/
def massive_accident_frequency : ℕ := 25

/-- Calculates the number of accidents of a given type -/
def accidents_of_type (frequency : ℕ) : ℕ :=
  total_seconds / frequency

/-- Theorem stating the total number of accidents in 4 minutes -/
theorem total_accidents_in_four_minutes :
  accidents_of_type car_collision_frequency +
  accidents_of_type big_crash_frequency +
  accidents_of_type pile_up_frequency +
  accidents_of_type massive_accident_frequency = 139 := by
  sorry


end NUMINAMATH_CALUDE_total_accidents_in_four_minutes_l2686_268663


namespace NUMINAMATH_CALUDE_find_k_l2686_268689

/-- The function f(x) -/
def f (k a x : ℝ) : ℝ := 2*k + (k^3)*a - x

/-- The function g(x) -/
def g (k a x : ℝ) : ℝ := x^2 + f k a x

theorem find_k (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ k : ℝ,
    (∀ x : ℝ, f k a x = -f k a (-x)) ∧  -- f is odd
    (∃ x : ℝ, f k a x = 3) ∧  -- f = 3 for some x
    (∀ x : ℝ, x ≥ 2 → g k a x ≥ -2) ∧  -- g has minimum -2 on [2, +∞)
    (∃ x : ℝ, x ≥ 2 ∧ g k a x = -2) ∧  -- g achieves minimum -2 on [2, +∞)
    k = 1 :=
  sorry

end NUMINAMATH_CALUDE_find_k_l2686_268689


namespace NUMINAMATH_CALUDE_round_trip_distance_prove_round_trip_distance_l2686_268650

def boat_speed : ℝ := 9
def stream_speed : ℝ := 6
def total_time : ℝ := 68

theorem round_trip_distance : ℝ :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let distance := (total_time * downstream_speed * upstream_speed) / (downstream_speed + upstream_speed)
  170

theorem prove_round_trip_distance : round_trip_distance = 170 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_distance_prove_round_trip_distance_l2686_268650


namespace NUMINAMATH_CALUDE_f_range_l2686_268631

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((1 - x) / (1 + x))

theorem f_range : ∀ x : ℝ, f x = -3 * π / 4 ∨ f x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l2686_268631


namespace NUMINAMATH_CALUDE_total_bathing_suits_l2686_268623

theorem total_bathing_suits (men_suits women_suits : ℕ) 
  (h1 : men_suits = 14797) 
  (h2 : women_suits = 4969) : 
  men_suits + women_suits = 19766 := by
  sorry

end NUMINAMATH_CALUDE_total_bathing_suits_l2686_268623


namespace NUMINAMATH_CALUDE_super_ball_distance_l2686_268633

def initial_height : ℚ := 80
def rebound_factor : ℚ := 2/3
def num_bounces : ℕ := 4

def bounce_sequence (n : ℕ) : ℚ :=
  initial_height * (rebound_factor ^ n)

def total_distance : ℚ :=
  2 * (initial_height * (1 - rebound_factor^(num_bounces + 1)) / (1 - rebound_factor)) - initial_height

theorem super_ball_distance :
  total_distance = 11280/81 :=
sorry

end NUMINAMATH_CALUDE_super_ball_distance_l2686_268633


namespace NUMINAMATH_CALUDE_square_diagonal_characterization_l2686_268635

/-- A quadrilateral with vertices A, B, C, and D in 2D space. -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The diagonals of a quadrilateral. -/
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((q.C.1 - q.A.1, q.C.2 - q.A.2), (q.D.1 - q.B.1, q.D.2 - q.B.2))

/-- Check if two vectors are perpendicular. -/
def are_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Check if a vector bisects another vector. -/
def bisects (v w : ℝ × ℝ) : Prop :=
  v.1 = w.1 / 2 ∧ v.2 = w.2 / 2

/-- Check if two vectors have equal length. -/
def equal_length (v w : ℝ × ℝ) : Prop :=
  v.1^2 + v.2^2 = w.1^2 + w.2^2

/-- A square is a quadrilateral with all sides equal and all angles right angles. -/
def is_square (q : Quadrilateral) : Prop :=
  let (AC, BD) := diagonals q
  are_perpendicular AC BD ∧
  bisects AC BD ∧
  bisects BD AC ∧
  equal_length AC BD

theorem square_diagonal_characterization (q : Quadrilateral) :
  is_square q ↔
    let (AC, BD) := diagonals q
    are_perpendicular AC BD ∧
    bisects AC BD ∧
    bisects BD AC ∧
    equal_length AC BD :=
  sorry

end NUMINAMATH_CALUDE_square_diagonal_characterization_l2686_268635


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l2686_268667

theorem inequality_system_solution_range (m : ℝ) : 
  (∃! (a b : ℤ), (a ≠ b) ∧ 
    ((a : ℝ) > -2) ∧ ((a : ℝ) ≤ (m + 2) / 3) ∧
    ((b : ℝ) > -2) ∧ ((b : ℝ) ≤ (m + 2) / 3) ∧
    (∀ (x : ℤ), (x ≠ a ∧ x ≠ b) → 
      ¬((x : ℝ) > -2 ∧ (x : ℝ) ≤ (m + 2) / 3))) →
  (-2 : ℝ) ≤ m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l2686_268667


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l2686_268682

/-- The number of people sitting at the table -/
def total_people : ℕ := 12

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 4

/-- The number of chemistry majors -/
def chemistry_majors : ℕ := 3

/-- The probability of all math majors sitting consecutively -/
def prob_consecutive_math : ℚ := 1 / 66

theorem math_majors_consecutive_probability :
  (total_people : ℚ) / (total_people.choose math_majors) = prob_consecutive_math := by
  sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l2686_268682


namespace NUMINAMATH_CALUDE_picnic_bread_slices_l2686_268646

/-- Calculate the total number of bread slices needed for a picnic --/
theorem picnic_bread_slices :
  let total_people : ℕ := 6
  let pb_people : ℕ := 4
  let tuna_people : ℕ := 3
  let turkey_people : ℕ := 2
  let pb_sandwiches_per_person : ℕ := 2
  let tuna_sandwiches_per_person : ℕ := 3
  let turkey_sandwiches_per_person : ℕ := 1
  let pb_slices_per_sandwich : ℕ := 2
  let tuna_slices_per_sandwich : ℕ := 3
  let turkey_slices_per_sandwich : ℚ := 3/2

  let total_pb_sandwiches := pb_people * pb_sandwiches_per_person
  let total_tuna_sandwiches := tuna_people * tuna_sandwiches_per_person
  let total_turkey_sandwiches := turkey_people * turkey_sandwiches_per_person

  let total_pb_slices := total_pb_sandwiches * pb_slices_per_sandwich
  let total_tuna_slices := total_tuna_sandwiches * tuna_slices_per_sandwich
  let total_turkey_slices := (total_turkey_sandwiches : ℚ) * turkey_slices_per_sandwich

  (total_pb_slices : ℚ) + (total_tuna_slices : ℚ) + total_turkey_slices = 46
  := by sorry

end NUMINAMATH_CALUDE_picnic_bread_slices_l2686_268646


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l2686_268697

theorem hexagon_angle_measure (A N G L E S : ℝ) : 
  -- ANGLES is a hexagon
  A + N + G + L + E + S = 720 →
  -- ∠A ≅ ∠G ≅ ∠E
  A = G ∧ G = E →
  -- ∠N is supplementary to ∠S
  N + S = 180 →
  -- ∠L is a right angle
  L = 90 →
  -- The measure of ∠E is 150°
  E = 150 := by sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l2686_268697


namespace NUMINAMATH_CALUDE_expression_equals_two_l2686_268692

theorem expression_equals_two (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) :
  (2 * x^2 - x) / ((x + 1) * (x - 2)) - (4 + x) / ((x + 1) * (x - 2)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_two_l2686_268692


namespace NUMINAMATH_CALUDE_hannah_bought_three_sweatshirts_l2686_268643

/-- Represents the purchase of sweatshirts and T-shirts by Hannah -/
structure Purchase where
  sweatshirts : ℕ
  tshirts : ℕ
  sweatshirt_cost : ℕ
  tshirt_cost : ℕ
  total_spent : ℕ

/-- Hannah's specific purchase -/
def hannahs_purchase : Purchase where
  sweatshirts := 0  -- We'll prove this should be 3
  tshirts := 2
  sweatshirt_cost := 15
  tshirt_cost := 10
  total_spent := 65

/-- The theorem stating that Hannah bought 3 sweatshirts -/
theorem hannah_bought_three_sweatshirts :
  ∃ (p : Purchase), p.tshirts = 2 ∧ p.sweatshirt_cost = 15 ∧ p.tshirt_cost = 10 ∧ p.total_spent = 65 ∧ p.sweatshirts = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_hannah_bought_three_sweatshirts_l2686_268643


namespace NUMINAMATH_CALUDE_student_fraction_mistake_l2686_268699

theorem student_fraction_mistake (n : ℚ) (correct_fraction : ℚ) (student_fraction : ℚ) :
  n = 288 →
  correct_fraction = 5 / 16 →
  student_fraction * n = correct_fraction * n + 150 →
  student_fraction = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_student_fraction_mistake_l2686_268699


namespace NUMINAMATH_CALUDE_coprime_power_minus_one_divisible_l2686_268674

theorem coprime_power_minus_one_divisible
  (N₁ N₂ : ℕ+) (k : ℕ) 
  (h_coprime : Nat.Coprime N₁ N₂)
  (h_k : k = Nat.totient N₂) :
  N₂ ∣ (N₁^k - 1) :=
by sorry

end NUMINAMATH_CALUDE_coprime_power_minus_one_divisible_l2686_268674


namespace NUMINAMATH_CALUDE_function_composition_property_l2686_268678

theorem function_composition_property (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + g y) = 2 * x + y) :
  ∀ x y : ℝ, g (x + f y) = x / 2 + y := by
  sorry

end NUMINAMATH_CALUDE_function_composition_property_l2686_268678


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2686_268676

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) + 1

-- State the theorem
theorem fixed_point_on_line 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : f a (-1) = 2) 
  (b : ℝ) 
  (h4 : b * (-1) + 2 + 1 = 0) :
  b = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2686_268676


namespace NUMINAMATH_CALUDE_min_value_x_minus_inv_y_l2686_268609

theorem min_value_x_minus_inv_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 3 * x + y + 1 / x + 2 / y = 13 / 2) :
  x - 1 / y ≥ -1 / 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
  3 * x₀ + y₀ + 1 / x₀ + 2 / y₀ = 13 / 2 ∧ x₀ - 1 / y₀ = -1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_minus_inv_y_l2686_268609


namespace NUMINAMATH_CALUDE_smallest_multiples_sum_l2686_268693

/-- The smallest positive two-digit multiple of 5 -/
def c : ℕ := 10

/-- The smallest positive three-digit multiple of 6 -/
def d : ℕ := 102

theorem smallest_multiples_sum : c + d = 112 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiples_sum_l2686_268693


namespace NUMINAMATH_CALUDE_smallest_root_of_quadratic_l2686_268670

theorem smallest_root_of_quadratic (x : ℝ) :
  (12 * x^2 - 44 * x + 40 = 0) → (x ≥ 5/3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_root_of_quadratic_l2686_268670


namespace NUMINAMATH_CALUDE_total_items_eq_137256_l2686_268685

/-- The number of old women going to Rome -/
def num_women : ℕ := 7

/-- The number of mules each woman has -/
def mules_per_woman : ℕ := 7

/-- The number of bags each mule carries -/
def bags_per_mule : ℕ := 7

/-- The number of loaves each bag contains -/
def loaves_per_bag : ℕ := 7

/-- The number of knives each loaf contains -/
def knives_per_loaf : ℕ := 7

/-- The number of sheaths each knife is in -/
def sheaths_per_knife : ℕ := 7

/-- The total number of items -/
def total_items : ℕ := 
  num_women +
  (num_women * mules_per_woman) +
  (num_women * mules_per_woman * bags_per_mule) +
  (num_women * mules_per_woman * bags_per_mule * loaves_per_bag) +
  (num_women * mules_per_woman * bags_per_mule * loaves_per_bag * knives_per_loaf) +
  (num_women * mules_per_woman * bags_per_mule * loaves_per_bag * knives_per_loaf * sheaths_per_knife)

theorem total_items_eq_137256 : total_items = 137256 := by
  sorry

end NUMINAMATH_CALUDE_total_items_eq_137256_l2686_268685


namespace NUMINAMATH_CALUDE_adam_has_23_tattoos_l2686_268657

/-- Calculates the number of tattoos Adam has given Jason's tattoo configuration -/
def adam_tattoos (jason_arm_tattoos jason_leg_tattoos jason_arms jason_legs : ℕ) : ℕ :=
  2 * (jason_arm_tattoos * jason_arms + jason_leg_tattoos * jason_legs) + 3

/-- Proves that Adam has 23 tattoos given Jason's tattoo configuration -/
theorem adam_has_23_tattoos :
  adam_tattoos 2 3 2 2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_adam_has_23_tattoos_l2686_268657


namespace NUMINAMATH_CALUDE_john_notebooks_correct_l2686_268637

/-- The number of notebooks John bought -/
def notebooks : ℕ := 5

/-- The number of pages in each notebook -/
def pages_per_notebook : ℕ := 40

/-- The number of pages John uses per day -/
def pages_per_day : ℕ := 4

/-- The number of days the notebooks last -/
def days : ℕ := 50

/-- Theorem stating that the number of notebooks John bought is correct -/
theorem john_notebooks_correct : 
  notebooks * pages_per_notebook = pages_per_day * days := by
  sorry


end NUMINAMATH_CALUDE_john_notebooks_correct_l2686_268637


namespace NUMINAMATH_CALUDE_car_speed_problem_l2686_268687

theorem car_speed_problem (total_distance : ℝ) (first_leg_distance : ℝ) (first_leg_speed : ℝ) (average_speed : ℝ) :
  total_distance = 320 →
  first_leg_distance = 160 →
  first_leg_speed = 75 →
  average_speed = 77.4193548387097 →
  let second_leg_distance := total_distance - first_leg_distance
  let total_time := total_distance / average_speed
  let first_leg_time := first_leg_distance / first_leg_speed
  let second_leg_time := total_time - first_leg_time
  let second_leg_speed := second_leg_distance / second_leg_time
  second_leg_speed = 80 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2686_268687


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_product_l2686_268632

/-- Two lines in a 2D plane are perpendicular iff the product of their slopes is -1 -/
theorem perpendicular_lines_slope_product (k₁ k₂ l₁ l₂ : ℝ) (hk₁ : k₁ ≠ 0) (hk₂ : k₂ ≠ 0) :
  (∀ x y₁ y₂ : ℝ, y₁ = k₁ * x + l₁ ∧ y₂ = k₂ * x + l₂) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, y₁ = k₁ * x₁ + l₁ ∧ y₂ = k₂ * x₂ + l₂ → 
    ((x₂ - x₁) * (k₁ * x₁ + l₁ - (k₂ * x₂ + l₂)) + (x₂ - x₁) * (y₂ - y₁) = 0)) ↔
  k₁ * k₂ = -1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_product_l2686_268632


namespace NUMINAMATH_CALUDE_game_points_l2686_268660

/-- Given a game where two players earn 10 points for winning a round,
    if they play 8 matches and one player wins 3/4 of the matches,
    prove that the other player earns 20 points. -/
theorem game_points (total_matches : ℕ) (points_per_win : ℕ) 
  (winner_fraction : ℚ) (h1 : total_matches = 8) 
  (h2 : points_per_win = 10) (h3 : winner_fraction = 3/4) : 
  (total_matches - (winner_fraction * total_matches).num) * points_per_win = 20 :=
by sorry

end NUMINAMATH_CALUDE_game_points_l2686_268660


namespace NUMINAMATH_CALUDE_complex_to_exponential_l2686_268617

theorem complex_to_exponential : 
  let z : ℂ := 1 + Complex.I * Real.sqrt 3
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ r = 2 ∧ θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_to_exponential_l2686_268617


namespace NUMINAMATH_CALUDE_max_lateral_surface_area_cylinder_in_sphere_l2686_268621

/-- The maximum lateral surface area of a cylinder inscribed in a sphere -/
theorem max_lateral_surface_area_cylinder_in_sphere :
  ∀ (R r l : ℝ),
  R > 0 →
  r > 0 →
  l > 0 →
  (4 / 3) * Real.pi * R^3 = (32 / 3) * Real.pi →
  r^2 + (l / 2)^2 = R^2 →
  2 * Real.pi * r * l ≤ 8 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_max_lateral_surface_area_cylinder_in_sphere_l2686_268621


namespace NUMINAMATH_CALUDE_equation_solution_l2686_268652

theorem equation_solution : 
  ∀ x y : ℚ, 
  y = 3 * x → 
  (5 * y^2 + 2 * y + 3 = 3 * (9 * x^2 + y + 1)) → 
  (x = 0 ∨ x = 1/6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2686_268652


namespace NUMINAMATH_CALUDE_sin_x_sin_2x_integral_l2686_268680

theorem sin_x_sin_2x_integral (x : ℝ) :
  deriv (λ x => (1/2) * Real.sin x - (1/6) * Real.sin (3*x)) x = Real.sin x * Real.sin (2*x) := by
  sorry

end NUMINAMATH_CALUDE_sin_x_sin_2x_integral_l2686_268680


namespace NUMINAMATH_CALUDE_circle_chord_segments_l2686_268634

theorem circle_chord_segments (r : ℝ) (chord_length : ℝ) : 
  r = 6 → 
  chord_length = 10 → 
  ∃ (m n : ℝ), m + n = 2*r ∧ m*n = (chord_length/2)^2 ∧ 
  ((m = 6 + Real.sqrt 11 ∧ n = 6 - Real.sqrt 11) ∨ 
   (m = 6 - Real.sqrt 11 ∧ n = 6 + Real.sqrt 11)) :=
by sorry

end NUMINAMATH_CALUDE_circle_chord_segments_l2686_268634


namespace NUMINAMATH_CALUDE_fitness_center_member_ratio_l2686_268648

theorem fitness_center_member_ratio 
  (f m : ℕ) -- number of female and male members
  (avg_female : ℕ := 45) -- average age of female members
  (avg_male : ℕ := 30) -- average age of male members
  (avg_all : ℕ := 35) -- average age of all members
  (h : (f * avg_female + m * avg_male) / (f + m) = avg_all) : 
  f / m = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_fitness_center_member_ratio_l2686_268648


namespace NUMINAMATH_CALUDE_triangle_db_length_l2686_268659

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)
  (right_angle_ABC : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (right_angle_ADB : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0)
  (AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 19)
  (AD_length : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 4)

-- Theorem statement
theorem triangle_db_length (t : Triangle) : 
  Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2) = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_db_length_l2686_268659


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l2686_268656

/-- A rectangular prism with dimensions a, b, and c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ := 12

/-- Theorem: A rectangular prism has 12 pairs of parallel edges -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) :
  parallel_edge_pairs prism = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l2686_268656


namespace NUMINAMATH_CALUDE_alyssas_total_spent_l2686_268696

/-- The amount Alyssa paid for grapes in dollars -/
def grapes_cost : ℚ := 12.08

/-- The amount Alyssa was refunded for cherries in dollars -/
def cherries_refund : ℚ := 9.85

/-- The total amount Alyssa spent in dollars -/
def total_spent : ℚ := grapes_cost - cherries_refund

/-- Theorem stating that the total amount Alyssa spent is $2.23 -/
theorem alyssas_total_spent : total_spent = 2.23 := by
  sorry

end NUMINAMATH_CALUDE_alyssas_total_spent_l2686_268696


namespace NUMINAMATH_CALUDE_tens_digit_of_19_power_2023_l2686_268619

theorem tens_digit_of_19_power_2023 : ∃ n : ℕ, 19^2023 ≡ 50 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_19_power_2023_l2686_268619


namespace NUMINAMATH_CALUDE_line_equation_proof_l2686_268679

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a line passes through a point -/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (l : Line) (p : Point) (given_line : Line) :
  p.x = 2 ∧ p.y = -1 ∧
  given_line.a = 2 ∧ given_line.b = 3 ∧ given_line.c = -4 ∧
  l.passesThrough p ∧
  l.isParallelTo given_line →
  l.a = 2 ∧ l.b = 3 ∧ l.c = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2686_268679


namespace NUMINAMATH_CALUDE_work_completion_l2686_268622

/-- Represents the number of days it takes to complete the work -/
structure WorkDays where
  together : ℝ
  a_alone : ℝ
  initial_together : ℝ
  a_remaining : ℝ

/-- Given work completion rates, proves that 'a' worked alone for 9 days after 'b' left -/
theorem work_completion (w : WorkDays) 
  (h1 : w.together = 40)
  (h2 : w.a_alone = 12)
  (h3 : w.initial_together = 10) : 
  w.a_remaining = 9 := by
sorry

end NUMINAMATH_CALUDE_work_completion_l2686_268622


namespace NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l2686_268602

theorem greatest_x_quadratic_inequality :
  ∃ (x : ℝ), x^2 - 6*x + 8 ≤ 0 ∧
  ∀ (y : ℝ), y^2 - 6*y + 8 ≤ 0 → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l2686_268602


namespace NUMINAMATH_CALUDE_complex_sum_equals_two_l2686_268669

theorem complex_sum_equals_two (z : ℂ) (h : z^7 = 1) (h2 : z = Complex.exp (2 * Real.pi * Complex.I / 7)) : 
  (z^2 / (1 + z^3)) + (z^4 / (1 + z^6)) + (z^6 / (1 + z^9)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_two_l2686_268669
