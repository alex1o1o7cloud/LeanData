import Mathlib

namespace NUMINAMATH_CALUDE_alternating_sum_equals_neg_151_l166_16644

/-- The sum of the alternating sequence 1-2+3-4+...+100-101 -/
def alternating_sum : ℤ := 1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20 + 21 - 22 + 23 - 24 + 25 - 26 + 27 - 28 + 29 - 30 + 31 - 32 + 33 - 34 + 35 - 36 + 37 - 38 + 39 - 40 + 41 - 42 + 43 - 44 + 45 - 46 + 47 - 48 + 49 - 50 + 51 - 52 + 53 - 54 + 55 - 56 + 57 - 58 + 59 - 60 + 61 - 62 + 63 - 64 + 65 - 66 + 67 - 68 + 69 - 70 + 71 - 72 + 73 - 74 + 75 - 76 + 77 - 78 + 79 - 80 + 81 - 82 + 83 - 84 + 85 - 86 + 87 - 88 + 89 - 90 + 91 - 92 + 93 - 94 + 95 - 96 + 97 - 98 + 99 - 100 + 101

theorem alternating_sum_equals_neg_151 : alternating_sum = -151 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sum_equals_neg_151_l166_16644


namespace NUMINAMATH_CALUDE_x_plus_y_between_52_and_53_l166_16601

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the problem conditions
def problem_conditions (x y : ℝ) : Prop :=
  y = 4 * (floor x) + 2 ∧
  y = 5 * (floor (x - 3)) + 7 ∧
  ∀ n : ℤ, x ≠ n

-- Theorem statement
theorem x_plus_y_between_52_and_53 (x y : ℝ) 
  (h : problem_conditions x y) : 
  52 < x + y ∧ x + y < 53 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_between_52_and_53_l166_16601


namespace NUMINAMATH_CALUDE_race_heartbeats_l166_16697

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proves that the total number of heartbeats during the specified race is 28800 -/
theorem race_heartbeats :
  total_heartbeats 160 30 6 = 28800 := by
  sorry

#eval total_heartbeats 160 30 6

end NUMINAMATH_CALUDE_race_heartbeats_l166_16697


namespace NUMINAMATH_CALUDE_power_difference_equality_l166_16686

theorem power_difference_equality : 2^2014 - (-2)^2015 = 3 * 2^2014 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equality_l166_16686


namespace NUMINAMATH_CALUDE_first_day_over_1000_l166_16603

def fungi_count (n : ℕ) : ℕ := 4 * 3^n

theorem first_day_over_1000 : ∃ n : ℕ, fungi_count n > 1000 ∧ ∀ m : ℕ, m < n → fungi_count m ≤ 1000 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_first_day_over_1000_l166_16603


namespace NUMINAMATH_CALUDE_decreasing_interval_of_sine_function_l166_16635

/-- Given a function f(x) = 2sin(2x + φ) where 0 < φ < π/2 and f(0) = √3,
    prove that the decreasing interval of f(x) on [0, π] is [π/12, 7π/12]. -/
theorem decreasing_interval_of_sine_function (φ : Real) 
    (h1 : 0 < φ) (h2 : φ < π/2) 
    (f : Real → Real) 
    (hf : ∀ x, f x = 2 * Real.sin (2 * x + φ)) 
    (h3 : f 0 = Real.sqrt 3) :
    (Set.Icc (π/12 : Real) (7*π/12) : Set Real) = 
    {x ∈ Set.Icc (0 : Real) π | ∀ y ∈ Set.Icc (0 : Real) π, x < y → f y < f x} :=
  sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_sine_function_l166_16635


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l166_16618

theorem expression_simplification_and_evaluation :
  let f (a : ℝ) := a / (a - 1) + (a + 1) / (a^2 - 1)
  let g (a : ℝ) := (a + 1) / (a - 1)
  ∀ a : ℝ, a^2 - 1 ≠ 0 →
    f a = g a ∧
    (a = 0 → g a = -1) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l166_16618


namespace NUMINAMATH_CALUDE_max_value_of_expression_l166_16698

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) :
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 9 ∧ 
    (x - y)^2 + (y - z)^2 + (z - x)^2 ≥ (a - b)^2 + (b - c)^2 + (c - a)^2) ∧
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 9 → 
    (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ 27) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l166_16698


namespace NUMINAMATH_CALUDE_fraction_multiplication_equality_l166_16663

theorem fraction_multiplication_equality : (1 / 2) * (1 / 3) * (1 / 4) * (1 / 6) * 144 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_equality_l166_16663


namespace NUMINAMATH_CALUDE_existence_of_basis_vectors_l166_16669

-- Define the set of points
variable (n : ℕ)
variable (O : ℝ × ℝ)
variable (A : Fin n → ℝ × ℝ)

-- Define the distance condition
variable (h : ∀ (i j : Fin n), ∃ (m : ℕ), ‖A i - A j‖ = Real.sqrt m)
variable (h' : ∀ (i : Fin n), ∃ (m : ℕ), ‖A i - O‖ = Real.sqrt m)

-- The theorem to be proved
theorem existence_of_basis_vectors :
  ∃ (x y : ℝ × ℝ), ∀ (i : Fin n), ∃ (k l : ℤ), A i - O = k • x + l • y :=
sorry

end NUMINAMATH_CALUDE_existence_of_basis_vectors_l166_16669


namespace NUMINAMATH_CALUDE_current_average_score_l166_16672

/-- Represents the bonus calculation and test scores for Karen's class -/
structure TestScores where
  baseBonus : ℕ := 500
  bonusPerPoint : ℕ := 10
  baseScore : ℕ := 75
  maxScore : ℕ := 150
  gradedTests : ℕ := 8
  totalTests : ℕ := 10
  targetBonus : ℕ := 600
  lastTwoTestsScore : ℕ := 290

/-- The theorem states that given the conditions, the current average score of the graded tests is 70 -/
theorem current_average_score (ts : TestScores) : 
  (ts.targetBonus - ts.baseBonus) / ts.bonusPerPoint + ts.baseScore = 85 →
  ts.gradedTests * (((ts.targetBonus - ts.baseBonus) / ts.bonusPerPoint + ts.baseScore) * ts.totalTests - ts.lastTwoTestsScore) / ts.totalTests = 70 := by
  sorry

end NUMINAMATH_CALUDE_current_average_score_l166_16672


namespace NUMINAMATH_CALUDE_presidentAndCommittee_ten_l166_16616

/-- The number of ways to choose a president and a 3-person committee from a group of n people,
    where the president cannot be on the committee and the order of choosing committee members does not matter. -/
def presidentAndCommittee (n : ℕ) : ℕ :=
  n * (n - 1).choose 3

/-- Theorem stating that for a group of 10 people, there are 840 ways to choose a president
    and a 3-person committee under the given conditions. -/
theorem presidentAndCommittee_ten :
  presidentAndCommittee 10 = 840 := by
  sorry

end NUMINAMATH_CALUDE_presidentAndCommittee_ten_l166_16616


namespace NUMINAMATH_CALUDE_quadratic_coefficient_determination_l166_16660

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- Theorem: If a quadratic function has vertex at (2, 5) and passes through (1, 2), then a = -3 -/
theorem quadratic_coefficient_determination (q : QuadraticFunction) 
  (vertex : q.f 2 = 5) 
  (point : q.f 1 = 2) : 
  q.a = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_determination_l166_16660


namespace NUMINAMATH_CALUDE_choose_six_three_equals_twenty_l166_16652

theorem choose_six_three_equals_twenty : Nat.choose 6 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_choose_six_three_equals_twenty_l166_16652


namespace NUMINAMATH_CALUDE_expression_evaluation_l166_16673

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator := 5 + x * (2 + x) - 2^2
  let denominator := x - 2 + x^2
  numerator / denominator = 1 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l166_16673


namespace NUMINAMATH_CALUDE_grade_assignment_count_l166_16691

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of available grades -/
def num_grades : ℕ := 4

/-- Theorem stating that the number of ways to assign grades is 4^15 -/
theorem grade_assignment_count :
  (num_grades : ℕ) ^ num_students = 1073741824 := by sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l166_16691


namespace NUMINAMATH_CALUDE_summer_jolly_degrees_l166_16661

/-- The combined number of degrees for Summer and Jolly -/
def combined_degrees (summer_degrees : ℕ) (difference : ℕ) : ℕ :=
  summer_degrees + (summer_degrees - difference)

/-- Theorem: Given Summer has 150 degrees and 5 more degrees than Jolly,
    the combined number of degrees they both have is 295. -/
theorem summer_jolly_degrees :
  combined_degrees 150 5 = 295 := by
  sorry

end NUMINAMATH_CALUDE_summer_jolly_degrees_l166_16661


namespace NUMINAMATH_CALUDE_polynomial_characterization_l166_16636

-- Define a polynomial with real coefficients
def RealPolynomial := ℝ → ℝ

-- Define the condition for a, b, c
def TripleCondition (a b c : ℝ) : Prop := a * b + b * c + c * a = 0

-- Define the polynomial equality condition
def PolynomialCondition (P : RealPolynomial) : Prop :=
  ∀ a b c : ℝ, TripleCondition a b c →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

-- Define the quadratic-quartic polynomial form
def QuadraticQuarticForm (P : RealPolynomial) : Prop :=
  ∃ a₂ a₄ : ℝ, ∀ x : ℝ, P x = a₂ * x^2 + a₄ * x^4

-- The main theorem
theorem polynomial_characterization :
  ∀ P : RealPolynomial, PolynomialCondition P → QuadraticQuarticForm P :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l166_16636


namespace NUMINAMATH_CALUDE_sandwich_cost_l166_16624

theorem sandwich_cost (num_sandwiches num_drinks drink_cost total_cost : ℕ) 
  (h1 : num_sandwiches = 3)
  (h2 : num_drinks = 2)
  (h3 : drink_cost = 4)
  (h4 : total_cost = 26) :
  ∃ (sandwich_cost : ℕ), 
    sandwich_cost = 6 ∧ 
    num_sandwiches * sandwich_cost + num_drinks * drink_cost = total_cost := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_l166_16624


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_b_l166_16628

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (sum_of_angles : A + B + C = 180)

-- Define an isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Theorem statement
theorem isosceles_triangle_angle_b (t : Triangle) 
  (ext_angle_A : ℝ) 
  (h_ext_angle : ext_angle_A = 110) 
  (h_ext_prop : t.B + t.C = ext_angle_A) :
  IsIsosceles t → t.B = 70 ∨ t.B = 55 ∨ t.B = 40 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_b_l166_16628


namespace NUMINAMATH_CALUDE_passing_percentage_is_30_l166_16637

def max_marks : ℕ := 600
def student_marks : ℕ := 80
def fail_margin : ℕ := 100

def passing_percentage : ℚ :=
  (student_marks + fail_margin : ℚ) / max_marks * 100

theorem passing_percentage_is_30 : passing_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_is_30_l166_16637


namespace NUMINAMATH_CALUDE_parallel_line_and_plane_existence_l166_16653

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define a plane in 3D space
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

-- Define parallelism between lines
def parallel_lines (l1 l2 : Line3D) : Prop := sorry

-- Define parallelism between a line and a plane
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop := sorry

-- Define a point not on a line
def point_not_on_line (p : ℝ × ℝ × ℝ) (l : Line3D) : Prop := sorry

theorem parallel_line_and_plane_existence 
  (l : Line3D) (p : ℝ × ℝ × ℝ) (h : point_not_on_line p l) : 
  (∃! l' : Line3D, parallel_lines l l' ∧ l'.point = p) ∧ 
  (∃ f : ℕ → Plane3D, (∀ n : ℕ, parallel_line_plane l (f n) ∧ (f n).point = p) ∧ 
                      (∀ n m : ℕ, n ≠ m → f n ≠ f m)) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_and_plane_existence_l166_16653


namespace NUMINAMATH_CALUDE_sum_of_four_squares_equals_prime_multiple_l166_16662

theorem sum_of_four_squares_equals_prime_multiple (p : Nat) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (m : Nat) (x₁ x₂ x₃ x₄ : Int), 
    m < p ∧ 
    x₁^2 + x₂^2 + x₃^2 + x₄^2 = m * p ∧ 
    (∀ (m' : Nat) (y₁ y₂ y₃ y₄ : Int), 
      m' < p → 
      y₁^2 + y₂^2 + y₃^2 + y₄^2 = m' * p → 
      m ≤ m') ∧
    m = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_squares_equals_prime_multiple_l166_16662


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l166_16658

/-- The number of y-intercepts of the parabola x = 3y^2 - 4y + 2 -/
theorem parabola_y_intercepts :
  let f : ℝ → ℝ := fun y => 3 * y^2 - 4 * y + 2
  (∃ y, f y = 0) = false :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l166_16658


namespace NUMINAMATH_CALUDE_plot_length_l166_16641

/-- Given a rectangular plot with the specified conditions, prove that its length is 70 meters. -/
theorem plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 40 →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  perimeter = 2 * (length + breadth) →
  total_cost = cost_per_meter * perimeter →
  length = 70 := by sorry

end NUMINAMATH_CALUDE_plot_length_l166_16641


namespace NUMINAMATH_CALUDE_solve_equation_l166_16614

theorem solve_equation : ∃ x : ℝ, 5 * (x - 9) = 6 * (3 - 3 * x) + 6 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l166_16614


namespace NUMINAMATH_CALUDE_students_making_stars_l166_16609

theorem students_making_stars (total_stars : ℕ) (stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) :
  total_stars / stars_per_student = 124 := by
  sorry

end NUMINAMATH_CALUDE_students_making_stars_l166_16609


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l166_16623

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h : ∀ n, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence with a₂ = 3 and a₅ = 6 is 1. -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h₂ : seq.a 2 = 3)
  (h₅ : seq.a 5 = 6) :
  seq.d = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l166_16623


namespace NUMINAMATH_CALUDE_fair_coin_tosses_l166_16682

/-- 
Given a fair coin with probability 1/2 for each side, 
if the probability of landing on the same side n times is 1/16, 
then n must be 4.
-/
theorem fair_coin_tosses (n : ℕ) : 
  (1 / 2 : ℝ) ^ n = 1 / 16 → n = 4 := by sorry

end NUMINAMATH_CALUDE_fair_coin_tosses_l166_16682


namespace NUMINAMATH_CALUDE_fruit_shop_problem_l166_16631

theorem fruit_shop_problem (total_cost : ℕ) (total_profit : ℕ) 
  (lychee_cost : ℕ) (longan_cost : ℕ) (lychee_price : ℕ) (longan_price : ℕ) 
  (second_profit : ℕ) :
  total_cost = 3900 →
  total_profit = 1200 →
  lychee_cost = 120 →
  longan_cost = 100 →
  lychee_price = 150 →
  longan_price = 140 →
  second_profit = 960 →
  ∃ (lychee_boxes longan_boxes : ℕ) (discount_rate : ℚ),
    lychee_cost * lychee_boxes + longan_cost * longan_boxes = total_cost ∧
    (lychee_price - lychee_cost) * lychee_boxes + (longan_price - longan_cost) * longan_boxes = total_profit ∧
    lychee_boxes = 20 ∧
    longan_boxes = 15 ∧
    (lychee_price - lychee_cost) * lychee_boxes + 
      (longan_price * discount_rate - longan_cost) * (2 * longan_boxes) = second_profit ∧
    discount_rate = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_fruit_shop_problem_l166_16631


namespace NUMINAMATH_CALUDE_hannah_dog_food_theorem_l166_16649

/-- The amount of dog food Hannah needs to prepare daily for her three dogs -/
def total_dog_food (first_dog_food : ℝ) (second_dog_multiplier : ℝ) (third_dog_additional : ℝ) : ℝ :=
  first_dog_food + 
  (first_dog_food * second_dog_multiplier) + 
  (first_dog_food * second_dog_multiplier + third_dog_additional)

/-- Theorem stating that Hannah needs to prepare 10 cups of dog food daily -/
theorem hannah_dog_food_theorem : 
  total_dog_food 1.5 2 2.5 = 10 := by
  sorry

#eval total_dog_food 1.5 2 2.5

end NUMINAMATH_CALUDE_hannah_dog_food_theorem_l166_16649


namespace NUMINAMATH_CALUDE_systematic_sampling_seventh_group_l166_16622

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (m : Nat) (k : Nat) : Nat :=
  (m + k) % 10

/-- The population size -/
def populationSize : Nat := 100

/-- The number of groups -/
def numGroups : Nat := 10

/-- The size of each group -/
def groupSize : Nat := populationSize / numGroups

/-- The starting number of the k-th group -/
def groupStart (k : Nat) : Nat :=
  (k - 1) * groupSize

theorem systematic_sampling_seventh_group :
  ∀ m : Nat,
    m = 6 →
    ∃ n : Nat,
      n = 63 ∧
      n ≥ groupStart 7 ∧
      n < groupStart 7 + groupSize ∧
      n % 10 = systematicSample m 7 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_seventh_group_l166_16622


namespace NUMINAMATH_CALUDE_negative_three_squared_opposite_l166_16619

/-- Two real numbers are opposite if their sum is zero -/
def are_opposite (a b : ℝ) : Prop := a + b = 0

/-- Theorem stating that (-3)² and -3² are opposite numbers -/
theorem negative_three_squared_opposite : are_opposite ((-3)^2) (-3^2) := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_opposite_l166_16619


namespace NUMINAMATH_CALUDE_singleEliminationTournament_l166_16664

/-- Calculates the number of games required in a single-elimination tournament. -/
def gamesRequired (numTeams : ℕ) : ℕ := numTeams - 1

/-- Theorem: In a single-elimination tournament with 23 teams, 22 games are required to determine the winner. -/
theorem singleEliminationTournament :
  gamesRequired 23 = 22 := by sorry

end NUMINAMATH_CALUDE_singleEliminationTournament_l166_16664


namespace NUMINAMATH_CALUDE_price_per_apple_l166_16627

/-- Calculate the price per apple given the orchard layout, apple production, and total revenue -/
theorem price_per_apple (rows : ℕ) (columns : ℕ) (apples_per_tree : ℕ) (total_revenue : ℚ) : 
  rows = 3 → columns = 4 → apples_per_tree = 5 → total_revenue = 30 →
  total_revenue / (rows * columns * apples_per_tree) = 0.5 := by
sorry

end NUMINAMATH_CALUDE_price_per_apple_l166_16627


namespace NUMINAMATH_CALUDE_grade_multiplier_is_five_l166_16688

def grades : List ℕ := [2, 2, 2, 3, 3, 3, 3, 4, 5]
def total_reward : ℚ := 15

theorem grade_multiplier_is_five :
  let average_grade := (grades.sum : ℚ) / grades.length
  let multiplier := total_reward / average_grade
  multiplier = 5 := by sorry

end NUMINAMATH_CALUDE_grade_multiplier_is_five_l166_16688


namespace NUMINAMATH_CALUDE_trig_simplification_l166_16668

theorem trig_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l166_16668


namespace NUMINAMATH_CALUDE_sum_inequality_l166_16610

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_abc : a + b + c = 1) :
  (1 / (b * c + a + 1 / a)) + (1 / (a * c + b + 1 / b)) + (1 / (a * b + c + 1 / c)) ≤ 27 / 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l166_16610


namespace NUMINAMATH_CALUDE_hannah_running_difference_l166_16642

def monday_distance : ℕ := 9
def wednesday_distance : ℕ := 4816
def friday_distance : ℕ := 2095

theorem hannah_running_difference :
  (monday_distance * 1000) - (wednesday_distance + friday_distance) = 2089 := by
  sorry

end NUMINAMATH_CALUDE_hannah_running_difference_l166_16642


namespace NUMINAMATH_CALUDE_total_groom_time_is_210_l166_16655

/-- Time to groom a poodle in minutes -/
def poodle_groom_time : ℕ := 30

/-- Time to groom a terrier in minutes -/
def terrier_groom_time : ℕ := poodle_groom_time / 2

/-- Number of poodles to groom -/
def num_poodles : ℕ := 3

/-- Number of terriers to groom -/
def num_terriers : ℕ := 8

/-- Total grooming time for all dogs -/
def total_groom_time : ℕ := num_poodles * poodle_groom_time + num_terriers * terrier_groom_time

theorem total_groom_time_is_210 : total_groom_time = 210 := by
  sorry

end NUMINAMATH_CALUDE_total_groom_time_is_210_l166_16655


namespace NUMINAMATH_CALUDE_dress_discount_calculation_l166_16646

def shoe_discount_percent : ℚ := 40 / 100
def original_shoe_price : ℚ := 50
def number_of_shoes : ℕ := 2
def original_dress_price : ℚ := 100
def total_spent : ℚ := 140

theorem dress_discount_calculation :
  let discounted_shoe_price := original_shoe_price * (1 - shoe_discount_percent)
  let total_shoe_cost := discounted_shoe_price * number_of_shoes
  let dress_cost := total_spent - total_shoe_cost
  original_dress_price - dress_cost = 20 := by sorry

end NUMINAMATH_CALUDE_dress_discount_calculation_l166_16646


namespace NUMINAMATH_CALUDE_robot_return_distance_l166_16695

/-- A robot's walk pattern -/
structure RobotWalk where
  step_distance : ℝ
  turn_angle : ℝ

/-- The total angle turned by the robot -/
def total_angle (w : RobotWalk) (n : ℕ) : ℝ := n * w.turn_angle

/-- The distance walked by the robot -/
def total_distance (w : RobotWalk) (n : ℕ) : ℝ := n * w.step_distance

/-- Theorem: A robot walking 1m and turning left 45° each time will return to its starting point after 8 steps -/
theorem robot_return_distance (w : RobotWalk) (h1 : w.step_distance = 1) (h2 : w.turn_angle = 45) :
  ∃ n : ℕ, total_angle w n = 360 ∧ total_distance w n = 8 := by
  sorry

end NUMINAMATH_CALUDE_robot_return_distance_l166_16695


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l166_16640

theorem quadratic_roots_sum_product (p q : ℝ) (k : ℕ+) :
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0) →
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x + y = 2) →
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x * y = k) →
  p = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l166_16640


namespace NUMINAMATH_CALUDE_second_to_first_angle_ratio_l166_16607

def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180

theorem second_to_first_angle_ratio 
  (a b c : ℝ) 
  (h_triangle : Triangle a b c)
  (h_second_multiple : ∃ k : ℝ, b = k * a)
  (h_third : c = 2 * a - 12)
  (h_measures : a = 32 ∧ b = 96 ∧ c = 52) :
  b / a = 3 := by
sorry

end NUMINAMATH_CALUDE_second_to_first_angle_ratio_l166_16607


namespace NUMINAMATH_CALUDE_parallel_linear_functions_theorem_l166_16608

/-- Two linear functions with parallel graphs not parallel to coordinate axes -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b ∧ g x = a * x + c
  not_axis_parallel : ∀ (a b c : ℝ), (∀ x, f x = a * x + b ∧ g x = a * x + c) → a ≠ 0

/-- The condition that (f(x))^2 touches -6g(x) -/
def touches_neg_6g (p : ParallelLinearFunctions) : Prop :=
  ∃! x, (p.f x)^2 = -6 * p.g x

/-- The condition that (g(x))^2 touches Af(x) -/
def touches_Af (p : ParallelLinearFunctions) (A : ℝ) : Prop :=
  ∃! x, (p.g x)^2 = A * p.f x

/-- The main theorem -/
theorem parallel_linear_functions_theorem (p : ParallelLinearFunctions) 
  (h : touches_neg_6g p) : 
  ∀ A, touches_Af p A ↔ (A = 6 ∨ A = 0) := by
  sorry

end NUMINAMATH_CALUDE_parallel_linear_functions_theorem_l166_16608


namespace NUMINAMATH_CALUDE_blue_pill_cost_l166_16677

/-- Represents the cost of pills for a 21-day regimen --/
structure PillCost where
  blue : ℝ
  yellow : ℝ
  total : ℝ
  h1 : blue = yellow + 3
  h2 : 21 * (blue + yellow) = total

/-- The theorem stating the cost of a blue pill given the conditions --/
theorem blue_pill_cost (pc : PillCost) (h : pc.total = 882) : pc.blue = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_blue_pill_cost_l166_16677


namespace NUMINAMATH_CALUDE_johns_expenses_exceed_earnings_l166_16659

/-- Represents the percentage of John's earnings spent on each category -/
structure Expenses where
  rent : ℝ
  dishwasher : ℝ
  groceries : ℝ

/-- Calculates John's expenses based on the given conditions -/
def calculate_expenses (rent_percent : ℝ) : Expenses :=
  { rent := rent_percent,
    dishwasher := rent_percent - (0.3 * rent_percent),
    groceries := rent_percent + (0.15 * rent_percent) }

/-- Theorem stating that John's expenses exceed his earnings -/
theorem johns_expenses_exceed_earnings (rent_percent : ℝ) 
  (h1 : rent_percent = 0.4)  -- John spent 40% of his earnings on rent
  (h2 : rent_percent > 0)    -- Rent percentage is positive
  (h3 : rent_percent < 1)    -- Rent percentage is less than 100%
  : (calculate_expenses rent_percent).rent + 
    (calculate_expenses rent_percent).dishwasher + 
    (calculate_expenses rent_percent).groceries > 1 := by
  sorry

#check johns_expenses_exceed_earnings

end NUMINAMATH_CALUDE_johns_expenses_exceed_earnings_l166_16659


namespace NUMINAMATH_CALUDE_complex_arithmetic_calculation_l166_16670

theorem complex_arithmetic_calculation :
  let B : ℂ := 5 - 2*I
  let N : ℂ := -3 + 2*I
  let T : ℂ := 2*I
  let Q : ℝ := 3
  B - N + T - 2 * (Q : ℂ) = 2 - 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_calculation_l166_16670


namespace NUMINAMATH_CALUDE_farm_legs_count_l166_16654

/-- Represents the number of legs for each animal type -/
def legs_per_animal (animal : String) : Nat :=
  match animal with
  | "chicken" => 2
  | "buffalo" => 4
  | _ => 0

/-- Calculates the total number of legs in the farm -/
def total_legs (total_animals : Nat) (chickens : Nat) : Nat :=
  let buffalos := total_animals - chickens
  chickens * legs_per_animal "chicken" + buffalos * legs_per_animal "buffalo"

/-- Theorem: In a farm with 9 animals, including 5 chickens and the rest buffalos, there are 26 legs in total -/
theorem farm_legs_count : total_legs 9 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_farm_legs_count_l166_16654


namespace NUMINAMATH_CALUDE_sample_size_is_120_l166_16689

/-- Represents the sizes of three population groups -/
structure PopulationGroups where
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ

/-- Calculates the total sample size in a stratified sampling -/
def calculateSampleSize (groups : PopulationGroups) (samplesFromGroup3 : ℕ) : ℕ :=
  samplesFromGroup3 * (groups.group1 + groups.group2 + groups.group3) / groups.group3

/-- Theorem stating that the sample size is 120 under given conditions -/
theorem sample_size_is_120 (groups : PopulationGroups) (h1 : groups.group1 = 2400) 
    (h2 : groups.group2 = 3600) (h3 : groups.group3 = 6000) (samplesFromGroup3 : ℕ) 
    (h4 : samplesFromGroup3 = 60) : 
  calculateSampleSize groups samplesFromGroup3 = 120 := by
  sorry

#eval calculateSampleSize ⟨2400, 3600, 6000⟩ 60

end NUMINAMATH_CALUDE_sample_size_is_120_l166_16689


namespace NUMINAMATH_CALUDE_jaden_initial_cars_l166_16612

/-- The number of toy cars Jaden had initially -/
def initial_cars : ℕ := 14

/-- The number of cars Jaden bought -/
def bought_cars : ℕ := 28

/-- The number of cars Jaden received as gifts -/
def gift_cars : ℕ := 12

/-- The number of cars Jaden gave to his sister -/
def sister_cars : ℕ := 8

/-- The number of cars Jaden gave to his friend -/
def friend_cars : ℕ := 3

/-- The number of cars Jaden has left -/
def remaining_cars : ℕ := 43

theorem jaden_initial_cars :
  initial_cars + bought_cars + gift_cars - sister_cars - friend_cars = remaining_cars :=
by sorry

end NUMINAMATH_CALUDE_jaden_initial_cars_l166_16612


namespace NUMINAMATH_CALUDE_sin_negative_120_degrees_l166_16687

theorem sin_negative_120_degrees : Real.sin (-(120 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_120_degrees_l166_16687


namespace NUMINAMATH_CALUDE_haley_washing_machine_capacity_l166_16650

/-- The number of pieces of clothing Haley's washing machine can wash at a time -/
def washing_machine_capacity (total_clothes : ℕ) (num_loads : ℕ) : ℕ :=
  total_clothes / num_loads

theorem haley_washing_machine_capacity :
  let total_shirts : ℕ := 2
  let total_sweaters : ℕ := 33
  let total_clothes : ℕ := total_shirts + total_sweaters
  let num_loads : ℕ := 5
  washing_machine_capacity total_clothes num_loads = 7 := by
  sorry

end NUMINAMATH_CALUDE_haley_washing_machine_capacity_l166_16650


namespace NUMINAMATH_CALUDE_binomial_mode_is_four_l166_16648

/-- The number of trials in the binomial distribution -/
def n : ℕ := 20

/-- The probability of success in each trial -/
def p : ℝ := 0.2

/-- The binomial probability mass function -/
def binomialPMF (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem stating that 4 is the mode of the binomial distribution B(20, 0.2) -/
theorem binomial_mode_is_four :
  ∀ k : ℕ, k ≠ 4 → binomialPMF 4 ≥ binomialPMF k :=
sorry

end NUMINAMATH_CALUDE_binomial_mode_is_four_l166_16648


namespace NUMINAMATH_CALUDE_roots_property_l166_16693

theorem roots_property (a b : ℝ) : 
  (a^2 - a - 2 = 0) → (b^2 - b - 2 = 0) → (a - 1) * (b - 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_roots_property_l166_16693


namespace NUMINAMATH_CALUDE_paper_width_calculation_l166_16666

theorem paper_width_calculation (w : ℝ) : 
  (2 * w * 17 = 2 * 8.5 * 11 + 100) → w = 287 / 34 := by
  sorry

end NUMINAMATH_CALUDE_paper_width_calculation_l166_16666


namespace NUMINAMATH_CALUDE_annas_walking_challenge_l166_16671

/-- Anna's walking challenge in March -/
theorem annas_walking_challenge 
  (total_days : ℕ) 
  (daily_target : ℝ) 
  (days_passed : ℕ) 
  (distance_walked : ℝ) 
  (h1 : total_days = 31) 
  (h2 : daily_target = 5) 
  (h3 : days_passed = 16) 
  (h4 : distance_walked = 95) : 
  (total_days * daily_target - distance_walked) / (total_days - days_passed) = 4 := by
  sorry

end NUMINAMATH_CALUDE_annas_walking_challenge_l166_16671


namespace NUMINAMATH_CALUDE_unique_odd_number_with_remainder_l166_16617

theorem unique_odd_number_with_remainder : 
  ∃! n : ℕ, 30 < n ∧ n < 50 ∧ n % 2 = 1 ∧ n % 7 = 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_odd_number_with_remainder_l166_16617


namespace NUMINAMATH_CALUDE_max_power_under_500_l166_16694

theorem max_power_under_500 (a b : ℕ) (ha : a > 0) (hb : b > 1) (hab : a^b < 500) :
  (∀ (c d : ℕ), c > 0 → d > 1 → c^d < 500 → a^b ≥ c^d) →
  a = 22 ∧ b = 2 ∧ a + b = 24 :=
sorry

end NUMINAMATH_CALUDE_max_power_under_500_l166_16694


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l166_16606

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 → x ≠ 3 → x ≠ 4 →
  (x^3 - 4*x^2 + 5*x - 7) / ((x - 1)*(x - 2)*(x - 3)*(x - 4)) =
  5/6 / (x - 1) + (-5/2) / (x - 2) + 1/2 / (x - 3) + 13/6 / (x - 4) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l166_16606


namespace NUMINAMATH_CALUDE_min_odd_integers_l166_16656

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum_ab : a + b = 32)
  (sum_abcd : a + b + c + d = 47)
  (sum_abcdef : a + b + c + d + e + f = 66) :
  ∃ (odds : Finset ℤ), odds ⊆ {a, b, c, d, e, f} ∧ 
    odds.card = 2 ∧ 
    (∀ x ∈ odds, Odd x) ∧
    (∀ y ∈ {a, b, c, d, e, f} \ odds, Even y) :=
sorry

end NUMINAMATH_CALUDE_min_odd_integers_l166_16656


namespace NUMINAMATH_CALUDE_intersection_locus_is_circle_l166_16675

/-- The locus of intersection points of two parametric lines forms a circle -/
theorem intersection_locus_is_circle :
  ∀ (x y u : ℝ),
  (u * x - 3 * y - 2 * u = 0) →
  (2 * x - 3 * u * y + u = 0) →
  ∃ (center_x center_y radius : ℝ),
  (x - center_x)^2 + (y - center_y)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_locus_is_circle_l166_16675


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l166_16665

def initial_bottle_caps : ℕ := 6
def found_bottle_caps : ℕ := 22

theorem danny_bottle_caps :
  initial_bottle_caps + found_bottle_caps = 28 :=
by sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l166_16665


namespace NUMINAMATH_CALUDE_f_f_zero_l166_16630

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -2 * Real.exp x else Real.log x

theorem f_f_zero (x : ℝ) : f (f x) = 0 ↔ x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_f_f_zero_l166_16630


namespace NUMINAMATH_CALUDE_sequence_property_l166_16699

theorem sequence_property (x y z : ℝ) 
  (h1 : (4 * y) ^ 2 = (3 * x) * (5 * z))  -- Geometric sequence condition
  (h2 : 2 / y = 1 / x + 1 / z)            -- Arithmetic sequence condition
  : x / z + z / x = 34 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l166_16699


namespace NUMINAMATH_CALUDE_sum_of_first_four_terms_l166_16638

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_first_four_terms
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_8th : a 8 = 21)
  (h_9th : a 9 = 17)
  (h_10th : a 10 = 13) :
  (a 1) + (a 2) + (a 3) + (a 4) = 172 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_four_terms_l166_16638


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l166_16625

theorem rectangular_to_polar_conversion :
  ∃ (r : ℝ) (θ : ℝ),
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r * Real.cos θ = 2 ∧
    r * Real.sin θ = -2 * Real.sqrt 3 ∧
    r = 4 ∧
    θ = 5 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l166_16625


namespace NUMINAMATH_CALUDE_percentage_increase_60_to_80_l166_16602

/-- The percentage increase when a value changes from 60 to 80 -/
theorem percentage_increase_60_to_80 : 
  (80 - 60) / 60 * 100 = 100 / 3 := by sorry

end NUMINAMATH_CALUDE_percentage_increase_60_to_80_l166_16602


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l166_16684

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : (1 : ℚ) / x + (1 : ℚ) / y = 5)
  (h2 : (1 : ℚ) / x - (1 : ℚ) / y = -9) :
  x + y = -5/14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l166_16684


namespace NUMINAMATH_CALUDE_blueberries_per_box_l166_16680

/-- The number of blueberries in each blue box -/
def B : ℕ := sorry

/-- The number of strawberries in each red box -/
def S : ℕ := sorry

/-- The difference between strawberries in a red box and blueberries in a blue box is 12 -/
axiom diff_strawberries_blueberries : S - B = 12

/-- Replacing one blue box with one red box increases the difference between total strawberries and total blueberries by 76 -/
axiom replacement_difference : 2 * S = 76

/-- The number of blueberries in each blue box is 26 -/
theorem blueberries_per_box : B = 26 := by sorry

end NUMINAMATH_CALUDE_blueberries_per_box_l166_16680


namespace NUMINAMATH_CALUDE_a_perpendicular_b_l166_16633

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def isPerpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Vector a in ℝ² -/
def a : ℝ × ℝ := (1, -2)

/-- Vector b in ℝ² -/
def b : ℝ × ℝ := (2, 1)

/-- Theorem stating that vectors a and b are perpendicular -/
theorem a_perpendicular_b : isPerpendicular a b := by
  sorry

end NUMINAMATH_CALUDE_a_perpendicular_b_l166_16633


namespace NUMINAMATH_CALUDE_inequality_of_four_positive_reals_l166_16643

theorem inequality_of_four_positive_reals (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_sum : a + b + c + d = 3) :
  1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 ≤ 1 / (a * b * c * d)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_four_positive_reals_l166_16643


namespace NUMINAMATH_CALUDE_sqrt_15_times_sqrt_3_minus_4_between_2_and_3_l166_16676

theorem sqrt_15_times_sqrt_3_minus_4_between_2_and_3 :
  2 < Real.sqrt 15 * Real.sqrt 3 - 4 ∧ Real.sqrt 15 * Real.sqrt 3 - 4 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_times_sqrt_3_minus_4_between_2_and_3_l166_16676


namespace NUMINAMATH_CALUDE_broker_income_slump_l166_16611

/-- 
Proves that if a broker's income remains unchanged when the commission rate 
increases from 4% to 5%, then the percentage slump in business is 20%.
-/
theorem broker_income_slump (X : ℝ) (Y : ℝ) (h : X > 0) :
  (0.04 * X = 0.05 * Y) →  -- Income remains unchanged
  (Y / X = 0.8)            -- Percentage slump in business is 20%
  := by sorry

end NUMINAMATH_CALUDE_broker_income_slump_l166_16611


namespace NUMINAMATH_CALUDE_sunflower_height_l166_16679

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Converts inches to feet, rounding down -/
def inches_to_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem sunflower_height (sister_height_feet : ℕ) (sister_height_inches : ℕ) 
  (height_difference : ℕ) :
  sister_height_feet = 4 →
  sister_height_inches = 3 →
  height_difference = 21 →
  inches_to_feet (feet_inches_to_inches sister_height_feet sister_height_inches + height_difference) = 6 :=
by sorry

end NUMINAMATH_CALUDE_sunflower_height_l166_16679


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l166_16615

/-- A quadratic function with a vertex at (-1, -3) -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The vertex of the quadratic function is at (-1, -3) -/
def has_vertex (b c : ℝ) : Prop :=
  (∀ x, f b c x ≤ f b c (-1)) ∧ (f b c (-1) = -3)

/-- Theorem stating that b = -2 and c = -4 for the given quadratic function -/
theorem quadratic_coefficients :
  ∃ b c : ℝ, has_vertex b c ∧ b = -2 ∧ c = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l166_16615


namespace NUMINAMATH_CALUDE_neds_weekly_revenue_l166_16674

/-- Calculates the weekly revenue for Ned's left-handed mouse store -/
def calculate_weekly_revenue (normal_mouse_price : ℝ) (price_increase_percentage : ℝ) 
  (daily_sales : ℕ) (open_days_per_week : ℕ) : ℝ :=
  let left_handed_mouse_price := normal_mouse_price * (1 + price_increase_percentage)
  let daily_revenue := left_handed_mouse_price * daily_sales
  daily_revenue * open_days_per_week

/-- Theorem stating that Ned's weekly revenue is $15600 -/
theorem neds_weekly_revenue : 
  calculate_weekly_revenue 120 0.3 25 4 = 15600 := by
  sorry

#eval calculate_weekly_revenue 120 0.3 25 4

end NUMINAMATH_CALUDE_neds_weekly_revenue_l166_16674


namespace NUMINAMATH_CALUDE_max_positive_integers_l166_16639

theorem max_positive_integers (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) :
  ∃ (pos : Finset ℤ), pos ⊆ {a, b, c, d, e, f} ∧ pos.card ≤ 5 ∧
  (∀ x ∈ pos, x > 0) ∧
  (∀ pos' : Finset ℤ, pos' ⊆ {a, b, c, d, e, f} → (∀ x ∈ pos', x > 0) → pos'.card ≤ pos.card) :=
by sorry

end NUMINAMATH_CALUDE_max_positive_integers_l166_16639


namespace NUMINAMATH_CALUDE_max_value_of_f_l166_16600

/-- The function we're maximizing -/
def f (t : ℤ) : ℚ := (3^t - 2*t) * t / 9^t

/-- The theorem stating the maximum value of the function -/
theorem max_value_of_f :
  ∃ (max : ℚ), max = 1/8 ∧ ∀ (t : ℤ), f t ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l166_16600


namespace NUMINAMATH_CALUDE_mrs_hilt_apple_consumption_l166_16685

-- Define the rate of apple consumption
def apples_per_hour : ℕ := 10

-- Define the number of hours
def total_hours : ℕ := 6

-- Theorem to prove
theorem mrs_hilt_apple_consumption :
  apples_per_hour * total_hours = 60 :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_apple_consumption_l166_16685


namespace NUMINAMATH_CALUDE_simplify_fraction_l166_16620

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l166_16620


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l166_16629

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo 0 2, 
    ∀ y ∈ Set.Ioo 0 2, 
      x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l166_16629


namespace NUMINAMATH_CALUDE_pizza_median_theorem_l166_16632

/-- Represents the pizza sales data for a day -/
structure PizzaSalesData where
  total_slices : ℕ
  total_customers : ℕ
  min_slices_per_customer : ℕ

/-- Calculates the maximum possible median number of slices per customer -/
def max_possible_median (data : PizzaSalesData) : ℚ :=
  sorry

/-- Theorem stating the maximum possible median for the given scenario -/
theorem pizza_median_theorem (data : PizzaSalesData) 
  (h1 : data.total_slices = 310)
  (h2 : data.total_customers = 150)
  (h3 : data.min_slices_per_customer = 1) :
  max_possible_median data = 7 := by
  sorry

end NUMINAMATH_CALUDE_pizza_median_theorem_l166_16632


namespace NUMINAMATH_CALUDE_eight_in_C_l166_16667

def C : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

theorem eight_in_C : 8 ∈ C := by
  sorry

end NUMINAMATH_CALUDE_eight_in_C_l166_16667


namespace NUMINAMATH_CALUDE_line_equation_proof_l166_16613

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Checks if a point lies on a line -/
def passes_through (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- The given line y = 4x + 3 -/
def given_line : Line :=
  { slope := 4, intercept := 3 }

/-- The point (1, 1) -/
def point : (ℝ × ℝ) :=
  (1, 1)

theorem line_equation_proof :
  ∃ (l : Line),
    parallel l given_line ∧
    passes_through l point.1 point.2 ∧
    l.slope = 4 ∧
    l.intercept = -3 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l166_16613


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l166_16605

theorem complex_magnitude_problem (z : ℂ) (h : z = 3 + I) :
  Complex.abs (z^2 - 3*z) = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l166_16605


namespace NUMINAMATH_CALUDE_subtract_from_zero_is_additive_inverse_l166_16621

theorem subtract_from_zero_is_additive_inverse (a : ℚ) : 0 - a = -a := by sorry

end NUMINAMATH_CALUDE_subtract_from_zero_is_additive_inverse_l166_16621


namespace NUMINAMATH_CALUDE_median_length_half_side_l166_16678

/-- Prove that the length of a median in a triangle is half the length of its corresponding side. -/
theorem median_length_half_side {A B C : ℝ × ℝ} : 
  let M := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)  -- Midpoint of BC
  dist A M = (1/2) * dist B C := by
  sorry

end NUMINAMATH_CALUDE_median_length_half_side_l166_16678


namespace NUMINAMATH_CALUDE_no_perfect_square_133_base_n_l166_16696

/-- Represents a number in base n -/
def base_n (digits : List Nat) (n : Nat) : Nat :=
  digits.foldr (fun d acc => d + n * acc) 0

/-- Checks if a number is a perfect square -/
def is_perfect_square (m : Nat) : Prop :=
  ∃ k : Nat, k * k = m

theorem no_perfect_square_133_base_n :
  ¬∃ n : Nat, 5 ≤ n ∧ n ≤ 15 ∧ is_perfect_square (base_n [1, 3, 3] n) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_133_base_n_l166_16696


namespace NUMINAMATH_CALUDE_one_more_tile_possible_l166_16692

/-- Represents a checkerboard -/
structure Checkerboard :=
  (size : ℕ)

/-- Represents a T-shaped tile -/
structure TTile :=
  (squares_covered : ℕ)

/-- The number of squares that remain uncovered after placing T-tiles -/
def uncovered_squares (board : Checkerboard) (tiles : ℕ) (tile : TTile) : ℕ :=
  board.size ^ 2 - tiles * tile.squares_covered

/-- Theorem stating that one more T-tile can be placed on the checkerboard -/
theorem one_more_tile_possible (board : Checkerboard) (tiles : ℕ) (tile : TTile) :
  board.size = 100 →
  tiles = 800 →
  tile.squares_covered = 4 →
  uncovered_squares board tiles tile ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_one_more_tile_possible_l166_16692


namespace NUMINAMATH_CALUDE_car_speed_first_hour_l166_16626

/-- Given a car's speed in the second hour and its average speed over two hours,
    calculate its speed in the first hour. -/
theorem car_speed_first_hour (second_hour_speed : ℝ) (average_speed : ℝ) :
  second_hour_speed = 60 →
  average_speed = 77.5 →
  (second_hour_speed + (average_speed * 2 - second_hour_speed)) / 2 = average_speed →
  average_speed * 2 - second_hour_speed = 95 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_first_hour_l166_16626


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l166_16645

theorem more_girls_than_boys (total_students : ℕ) 
  (h_total : total_students = 42) 
  (h_ratio : ∃ (x : ℕ), 3 * x + 4 * x = total_students) : 
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    3 * girls = 4 * boys ∧ 
    girls = boys + 6 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l166_16645


namespace NUMINAMATH_CALUDE_triangle_base_calculation_l166_16683

theorem triangle_base_calculation (square_perimeter : ℝ) (triangle_area : ℝ) :
  square_perimeter = 60 →
  triangle_area = 150 →
  let square_side := square_perimeter / 4
  let triangle_height := square_side
  triangle_area = 1/2 * triangle_height * (triangle_base : ℝ) →
  triangle_base = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_calculation_l166_16683


namespace NUMINAMATH_CALUDE_plan_y_more_cost_effective_l166_16604

/-- Represents the cost of Plan X in cents for y gigabytes of data -/
def plan_x_cost (y : ℝ) : ℝ := 25 * y

/-- Represents the cost of Plan Y in cents for y gigabytes of data -/
def plan_y_cost (y : ℝ) : ℝ := 1500 + 15 * y

/-- The minimum number of gigabytes for Plan Y to be more cost-effective -/
def min_gb_for_plan_y : ℝ := 150

theorem plan_y_more_cost_effective :
  ∀ y : ℝ, y ≥ min_gb_for_plan_y → plan_y_cost y < plan_x_cost y :=
by sorry

end NUMINAMATH_CALUDE_plan_y_more_cost_effective_l166_16604


namespace NUMINAMATH_CALUDE_binary_product_in_base4_l166_16647

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The first binary number: 1101₂ -/
def binary1 : List Bool := [true, true, false, true]

/-- The second binary number: 111₂ -/
def binary2 : List Bool := [true, true, true]

/-- Statement: The product of 1101₂ and 111₂ in base 4 is 311₄ -/
theorem binary_product_in_base4 :
  decimal_to_base4 (binary_to_decimal binary1 * binary_to_decimal binary2) = [3, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_product_in_base4_l166_16647


namespace NUMINAMATH_CALUDE_x_range_l166_16634

/-- The function f(x) = x^2 + ax -/
def f (x a : ℝ) : ℝ := x^2 + a*x

/-- The theorem stating the range of x given the conditions -/
theorem x_range (x : ℝ) :
  (∀ a ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x a ≥ 3 - a) →
  (x ≤ -1 - Real.sqrt 2 ∨ x ≥ 1 + Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_x_range_l166_16634


namespace NUMINAMATH_CALUDE_clerical_percentage_theorem_l166_16651

/-- Represents the employee composition of a company -/
structure CompanyEmployees where
  total : ℕ
  clerical_ratio : ℚ
  management_ratio : ℚ
  clerical_reduction : ℚ

/-- Calculates the percentage of clerical employees after reduction -/
def clerical_percentage_after_reduction (c : CompanyEmployees) : ℚ :=
  let initial_clerical := c.clerical_ratio * c.total
  let reduced_clerical := initial_clerical - c.clerical_reduction * initial_clerical
  let total_after_reduction := c.total - (initial_clerical - reduced_clerical)
  (reduced_clerical / total_after_reduction) * 100

/-- Theorem stating the result of the employee reduction -/
theorem clerical_percentage_theorem (c : CompanyEmployees) 
  (h1 : c.total = 5000)
  (h2 : c.clerical_ratio = 3/7)
  (h3 : c.management_ratio = 1/3)
  (h4 : c.clerical_reduction = 3/8) :
  ∃ (ε : ℚ), abs (clerical_percentage_after_reduction c - 3194/100) < ε ∧ ε < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_clerical_percentage_theorem_l166_16651


namespace NUMINAMATH_CALUDE_repeating_decimal_56_equals_fraction_l166_16657

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

theorem repeating_decimal_56_equals_fraction :
  RepeatingDecimal 5 6 = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_56_equals_fraction_l166_16657


namespace NUMINAMATH_CALUDE_basketball_probabilities_l166_16690

def probability_A : ℝ := 0.7
def shots : ℕ := 3

theorem basketball_probabilities (a : ℝ) 
  (h1 : 0 ≤ a ∧ a ≤ 1) 
  (h2 : (Nat.choose 3 2 : ℝ) * (1 - probability_A) * probability_A^2 + probability_A^3 - a^3 = 0.659) :
  a = 0.5 ∧ 
  (1 - probability_A)^3 * (1 - a)^3 + 
  (Nat.choose 3 1 : ℝ) * (1 - probability_A)^2 * probability_A * 
  (Nat.choose 3 1 : ℝ) * (1 - a)^2 * a = 0.07425 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probabilities_l166_16690


namespace NUMINAMATH_CALUDE_percent_difference_l166_16681

theorem percent_difference (N M : ℝ) (h : N > 0) : 
  let N' := 1.5 * N
  100 - (M / N') * 100 = 100 - (200 * M) / (3 * N) :=
by sorry

end NUMINAMATH_CALUDE_percent_difference_l166_16681
