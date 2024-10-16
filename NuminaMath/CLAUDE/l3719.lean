import Mathlib

namespace NUMINAMATH_CALUDE_linear_equation_result_l3719_371956

theorem linear_equation_result (m x : ℝ) : 
  (m^2 - 1 = 0) → 
  (m - 1 ≠ 0) → 
  ((m^2 - 1)*x^2 - (m - 1)*x - 8 = 0) →
  200*(x - m)*(x + 2*m) - 10*m = 2010 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_result_l3719_371956


namespace NUMINAMATH_CALUDE_total_shaded_area_specific_total_shaded_area_l3719_371983

/-- The total shaded area of three overlapping rectangles -/
theorem total_shaded_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ)
  (shared_side triple_overlap_width : ℕ) : ℕ :=
  let rect1_area := rect1_width * rect1_height
  let rect2_area := rect2_width * rect2_height
  let rect3_area := rect3_width * rect3_height
  let overlap_area := shared_side * shared_side
  let triple_overlap_area := triple_overlap_width * shared_side
  rect1_area + rect2_area + rect3_area - overlap_area - triple_overlap_area

/-- The total shaded area of the specific configuration is 136 square units -/
theorem specific_total_shaded_area :
  total_shaded_area 4 15 5 10 3 18 4 3 = 136 := by
  sorry

end NUMINAMATH_CALUDE_total_shaded_area_specific_total_shaded_area_l3719_371983


namespace NUMINAMATH_CALUDE_car_selling_price_l3719_371966

/-- Calculates the selling price of a car given its purchase price, repair costs, and profit percentage. -/
def selling_price (purchase_price repair_costs profit_percent : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_costs
  let profit := (profit_percent / 100) * total_cost
  total_cost + profit

/-- Theorem stating that for the given conditions, the selling price is 64900. -/
theorem car_selling_price :
  selling_price 42000 8000 29.8 = 64900 := by
  sorry

end NUMINAMATH_CALUDE_car_selling_price_l3719_371966


namespace NUMINAMATH_CALUDE_unsatisfactory_fraction_is_8_25_l3719_371942

/-- Represents the grades in a class -/
structure GradeDistribution where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  f : Nat

/-- The grade distribution for the given class -/
def classGrades : GradeDistribution :=
  { a := 6, b := 5, c := 4, d := 2, f := 8 }

/-- The total number of students in the class -/
def totalStudents (grades : GradeDistribution) : Nat :=
  grades.a + grades.b + grades.c + grades.d + grades.f

/-- The number of students with unsatisfactory grades -/
def unsatisfactoryGrades (grades : GradeDistribution) : Nat :=
  grades.f

/-- Theorem: The fraction of unsatisfactory grades is 8/25 -/
theorem unsatisfactory_fraction_is_8_25 :
  (unsatisfactoryGrades classGrades : Rat) / (totalStudents classGrades) = 8 / 25 := by
  sorry

end NUMINAMATH_CALUDE_unsatisfactory_fraction_is_8_25_l3719_371942


namespace NUMINAMATH_CALUDE_perception_arrangements_l3719_371906

/-- The number of distinct arrangements of letters in a word with specific letter frequencies -/
def word_arrangements (total : ℕ) (double_count : ℕ) (single_count : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial 2 ^ double_count)

/-- Theorem stating the number of arrangements for the given word structure -/
theorem perception_arrangements :
  word_arrangements 10 3 4 = 453600 := by
  sorry

#eval word_arrangements 10 3 4

end NUMINAMATH_CALUDE_perception_arrangements_l3719_371906


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achieved_l3719_371946

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (1 / (x + y)) + ((x + y) / z) ≥ 3 := by
  sorry

theorem min_value_achieved (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ 
  x' + y' + z' = 1 ∧ 
  (1 / (x' + y')) + ((x' + y') / z') = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achieved_l3719_371946


namespace NUMINAMATH_CALUDE_vector_operation_proof_l3719_371973

theorem vector_operation_proof :
  (4 : ℝ) • (![3, -5] : Fin 2 → ℝ) - (3 : ℝ) • (![2, -6] : Fin 2 → ℝ) = ![6, -2] := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l3719_371973


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l3719_371968

theorem floor_ceiling_sum : ⌊(-0.123 : ℝ)⌋ + ⌈(4.567 : ℝ)⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l3719_371968


namespace NUMINAMATH_CALUDE_array_sum_remainder_l3719_371940

/-- Represents the sum of all terms in a 1/1004-array -/
def array_sum : ℚ := (2 * 1004^2) / ((2 * 1004 - 1) * (1004 - 1))

/-- Numerator of the array sum when expressed in lowest terms -/
def m : ℕ := 2 * 1004^2

/-- Denominator of the array sum when expressed in lowest terms -/
def n : ℕ := (2 * 1004 - 1) * (1004 - 1)

/-- The main theorem stating that (m + n) ≡ 0 (mod 1004) -/
theorem array_sum_remainder :
  (m + n) % 1004 = 0 := by sorry

end NUMINAMATH_CALUDE_array_sum_remainder_l3719_371940


namespace NUMINAMATH_CALUDE_sum_of_s_coordinates_l3719_371916

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle defined by four points -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Given a rectangle PQRS with P and R as diagonally opposite corners,
    proves that the sum of coordinates of S is 8 -/
theorem sum_of_s_coordinates (rect : Rectangle) : 
  rect.P = Point.mk (-3) (-2) →
  rect.R = Point.mk 9 1 →
  rect.Q = Point.mk 2 (-5) →
  rect.S.x + rect.S.y = 8 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_s_coordinates_l3719_371916


namespace NUMINAMATH_CALUDE_special_sequence_sixth_term_l3719_371958

/-- A sequence of positive integers where each term after the first is 1/3 of the sum of the term that precedes it and the term that follows it. -/
def SpecialSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → a n > 0 ∧ a (n + 1) = (a n + a (n + 2)) / 3

theorem special_sequence_sixth_term
  (a : ℕ → ℚ)
  (h_seq : SpecialSequence a)
  (h_first : a 1 = 3)
  (h_fifth : a 5 = 54) :
  a 6 = 1133 / 7 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_sixth_term_l3719_371958


namespace NUMINAMATH_CALUDE_lemon_heads_distribution_l3719_371972

def small_package : ℕ := 6
def medium_package : ℕ := 15
def large_package : ℕ := 30

def louis_small_packages : ℕ := 5
def louis_medium_packages : ℕ := 3
def louis_large_packages : ℕ := 2

def louis_eaten : ℕ := 54
def num_friends : ℕ := 4

theorem lemon_heads_distribution :
  let total := louis_small_packages * small_package + 
               louis_medium_packages * medium_package + 
               louis_large_packages * large_package
  let remaining := total - louis_eaten
  let per_friend := remaining / num_friends
  per_friend = 3 * small_package + 2 ∧ 
  remaining % num_friends = 1 := by sorry

end NUMINAMATH_CALUDE_lemon_heads_distribution_l3719_371972


namespace NUMINAMATH_CALUDE_gcd_21_and_number_between_50_60_l3719_371994

theorem gcd_21_and_number_between_50_60 :
  ∃! n : ℕ, 50 ≤ n ∧ n ≤ 60 ∧ Nat.gcd 21 n = 7 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_gcd_21_and_number_between_50_60_l3719_371994


namespace NUMINAMATH_CALUDE_lindsay_squat_weight_l3719_371933

/-- The total weight Lindsey will squat -/
def total_weight (num_bands : ℕ) (resistance_per_band : ℕ) (dumbbell_weight : ℕ) : ℕ :=
  num_bands * resistance_per_band + dumbbell_weight

/-- Theorem stating the total weight Lindsey will squat -/
theorem lindsay_squat_weight :
  let num_bands : ℕ := 2
  let resistance_per_band : ℕ := 5
  let dumbbell_weight : ℕ := 10
  total_weight num_bands resistance_per_band dumbbell_weight = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_lindsay_squat_weight_l3719_371933


namespace NUMINAMATH_CALUDE_smallest_cube_multiple_l3719_371995

theorem smallest_cube_multiple : 
  (∃ (x : ℕ+) (M : ℤ), 3960 * x.val = M^3) ∧ 
  (∀ (y : ℕ+) (N : ℤ), 3960 * y.val = N^3 → y.val ≥ 9075) := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_multiple_l3719_371995


namespace NUMINAMATH_CALUDE_book_page_digits_l3719_371931

/-- The total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  (min n 9) + 
  (2 * (min n 99 - 9)) + 
  (3 * (n - min n 99))

/-- Theorem: The total number of digits used in numbering the pages of a book with 346 pages is 930 -/
theorem book_page_digits : totalDigits 346 = 930 := by
  sorry

end NUMINAMATH_CALUDE_book_page_digits_l3719_371931


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3719_371957

/-- A point in the second quadrant with |x| = 2 and |y| = 3 -/
structure PointM where
  x : ℝ
  y : ℝ
  second_quadrant : x < 0 ∧ y > 0
  abs_x_eq_two : |x| = 2
  abs_y_eq_three : |y| = 3

/-- The coordinates of a point symmetric to M with respect to the y-axis -/
def symmetric_point (m : PointM) : ℝ × ℝ := (-m.x, m.y)

theorem symmetric_point_coordinates (m : PointM) : 
  symmetric_point m = (2, 3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3719_371957


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l3719_371964

theorem triangle_angle_sum (X Y Z : ℝ) (h1 : X + Y = 80) (h2 : X + Y + Z = 180) : Z = 100 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l3719_371964


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l3719_371935

theorem smallest_divisible_by_1_to_12 : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ m) → n ≤ m) ∧ n = 27720 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l3719_371935


namespace NUMINAMATH_CALUDE_subtract_negative_l3719_371970

theorem subtract_negative : -2 - (-3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l3719_371970


namespace NUMINAMATH_CALUDE_line_equation_through_parabola_intersection_l3719_371989

/-- The equation of a line passing through (0, 2) and intersecting the parabola y² = 2x
    at two points M and N, where OM · ON = 0, is x + y - 2 = 0 -/
theorem line_equation_through_parabola_intersection (x y : ℝ) :
  let parabola := (fun (x y : ℝ) ↦ y^2 = 2*x)
  let line := (fun (x y : ℝ) ↦ ∃ (k : ℝ), y = k*x + 2)
  let O := (0, 0)
  let M := (x, y)
  let N := (2/y, y)  -- Using the parabola equation to express N
  parabola x y ∧
  line x y ∧
  (M.1 * N.1 + M.2 * N.2 = 0)  -- OM · ON = 0
  →
  x + y - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_parabola_intersection_l3719_371989


namespace NUMINAMATH_CALUDE_three_digit_numbers_satisfying_condition_l3719_371962

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def abc_to_number (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

def bca_to_number (a b c : ℕ) : ℕ :=
  100 * b + 10 * c + a

def cab_to_number (a b c : ℕ) : ℕ :=
  100 * c + 10 * a + b

def satisfies_condition (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    n = abc_to_number a b c ∧
    2 * n = bca_to_number a b c + cab_to_number a b c

theorem three_digit_numbers_satisfying_condition :
  {n : ℕ | is_three_digit_number n ∧ satisfies_condition n} = {481, 518, 592, 629} :=
sorry

end NUMINAMATH_CALUDE_three_digit_numbers_satisfying_condition_l3719_371962


namespace NUMINAMATH_CALUDE_quadratic_polynomial_solution_l3719_371908

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  is_quadratic : a ≠ 0

/-- Evaluation of a quadratic polynomial at a point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem quadratic_polynomial_solution 
  (f g : QuadraticPolynomial) 
  (h1 : ∀ x, f.eval (g.eval x) = (f.eval x) * (g.eval x))
  (h2 : g.eval 3 = 40) :
  g.a = 1 ∧ g.b = 31/2 ∧ g.c = -31/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_solution_l3719_371908


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_7560_l3719_371936

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  sorry

theorem largest_perfect_square_factor_7560 :
  largest_perfect_square_factor 7560 = 36 := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_7560_l3719_371936


namespace NUMINAMATH_CALUDE_largest_circle_equation_l3719_371990

/-- The line equation ax - y - 4a - 2 = 0, where a is a real number -/
def line_equation (a x y : ℝ) : Prop := a * x - y - 4 * a - 2 = 0

/-- The center of the circle is at point (2, 0) -/
def circle_center : ℝ × ℝ := (2, 0)

/-- The equation of a circle with center (h, k) and radius r -/
def circle_equation (x y h k r : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

theorem largest_circle_equation :
  ∃ (r : ℝ), r > 0 ∧
  (∀ a : ℝ, ∃ x y : ℝ, line_equation a x y ∧ circle_equation x y circle_center.1 circle_center.2 r) ∧
  (∀ r' : ℝ, r' > 0 →
    (∀ a : ℝ, ∃ x y : ℝ, line_equation a x y ∧ circle_equation x y circle_center.1 circle_center.2 r') →
    r' ≤ r) ∧
  (∀ x y : ℝ, circle_equation x y circle_center.1 circle_center.2 r ↔ (x - 2)^2 + y^2 = 8) :=
sorry

end NUMINAMATH_CALUDE_largest_circle_equation_l3719_371990


namespace NUMINAMATH_CALUDE_fish_tank_problem_l3719_371939

theorem fish_tank_problem (tank1_size : ℚ) (tank2_size : ℚ) (tank1_water : ℚ) 
  (fish2_length : ℚ) (fish_diff : ℕ) :
  tank1_size = 2 * tank2_size →
  tank1_water = 48 →
  fish2_length = 2 →
  fish_diff = 3 →
  ∃ (fish1_length : ℚ),
    fish1_length = 3 ∧
    (tank1_water / fish1_length - 1 = tank2_size / fish2_length + fish_diff) :=
by sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l3719_371939


namespace NUMINAMATH_CALUDE_inequality_range_l3719_371923

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l3719_371923


namespace NUMINAMATH_CALUDE_star_3_5_l3719_371907

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 + 3*a*b + b^2

-- Theorem statement
theorem star_3_5 : star 3 5 = 79 := by sorry

end NUMINAMATH_CALUDE_star_3_5_l3719_371907


namespace NUMINAMATH_CALUDE_three_digit_cube_ending_777_l3719_371905

theorem three_digit_cube_ending_777 :
  ∃! x : ℕ, 100 ≤ x ∧ x < 1000 ∧ x^3 % 1000 = 777 :=
by
  use 753
  sorry

end NUMINAMATH_CALUDE_three_digit_cube_ending_777_l3719_371905


namespace NUMINAMATH_CALUDE_reading_time_proof_l3719_371951

/-- Calculates the number of weeks needed to read a series of books -/
def weeks_to_read (total_books : ℕ) (first_week : ℕ) (subsequent_weeks : ℕ) : ℕ :=
  1 + (total_books - first_week + subsequent_weeks - 1) / subsequent_weeks

/-- Proves that reading 70 books takes 11 weeks when reading 5 books in the first week and 7 books per week thereafter -/
theorem reading_time_proof :
  weeks_to_read 70 5 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_proof_l3719_371951


namespace NUMINAMATH_CALUDE_profit_percentage_l3719_371986

theorem profit_percentage (C S : ℝ) (h : C > 0) : 
  20 * C = 16 * S → (S - C) / C * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l3719_371986


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_l3719_371917

/-- Given five consecutive odd numbers, prove that if the sum of the first and third is 146, then the fifth number is 79 -/
theorem consecutive_odd_numbers (a b c d e : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  b = a + 2 →               -- b is the next odd number after a
  c = a + 4 →               -- c is the next odd number after b
  d = a + 6 →               -- d is the next odd number after c
  e = a + 8 →               -- e is the next odd number after d
  a + c = 146 →             -- sum of a and c is 146
  e = 79 := by              -- prove that e equals 79
sorry


end NUMINAMATH_CALUDE_consecutive_odd_numbers_l3719_371917


namespace NUMINAMATH_CALUDE_simplify_expression_l3719_371920

theorem simplify_expression (m : ℝ) (h : m ≠ 3) : 
  (m^2 / (m-3)) + (9 / (3-m)) = m + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3719_371920


namespace NUMINAMATH_CALUDE_solution_set_inequalities_l3719_371961

theorem solution_set_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequalities_l3719_371961


namespace NUMINAMATH_CALUDE_mhsc_unanswered_questions_l3719_371979

/-- Represents the scoring system for the Math High School Contest -/
structure ScoringSystem where
  initial : ℤ
  correct : ℤ
  wrong : ℤ
  unanswered : ℤ

/-- Calculates the score based on a given scoring system and number of questions -/
def calculateScore (system : ScoringSystem) (correct wrong unanswered : ℕ) : ℤ :=
  system.initial + system.correct * correct + system.wrong * wrong + system.unanswered * unanswered

theorem mhsc_unanswered_questions (newSystem oldSystem : ScoringSystem)
    (totalQuestions newScore oldScore : ℕ) :
    newSystem = ScoringSystem.mk 0 6 0 1 →
    oldSystem = ScoringSystem.mk 25 5 (-2) 0 →
    totalQuestions = 30 →
    newScore = 110 →
    oldScore = 95 →
    ∃ (correct wrong unanswered : ℕ),
      correct + wrong + unanswered = totalQuestions ∧
      calculateScore newSystem correct wrong unanswered = newScore ∧
      calculateScore oldSystem correct wrong unanswered = oldScore ∧
      unanswered = 10 :=
  sorry


end NUMINAMATH_CALUDE_mhsc_unanswered_questions_l3719_371979


namespace NUMINAMATH_CALUDE_population_scientific_notation_l3719_371944

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem population_scientific_notation :
  toScientificNotation (141260 * 1000000) =
    ScientificNotation.mk 1.4126 5 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_population_scientific_notation_l3719_371944


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_solid_l3719_371918

theorem sphere_volume_circumscribing_rectangular_solid :
  let length : ℝ := 2
  let width : ℝ := 1
  let height : ℝ := 2
  let space_diagonal : ℝ := Real.sqrt (length^2 + width^2 + height^2)
  let sphere_radius : ℝ := space_diagonal / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius^3
  sphere_volume = (9 / 2) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_solid_l3719_371918


namespace NUMINAMATH_CALUDE_smallest_m_is_one_l3719_371950

/-- The largest prime with 2023 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2023 digits -/
axiom q_digits : q ≥ 10^2022 ∧ q < 10^2023

/-- q is the largest prime with 2023 digits -/
axiom q_largest : ∀ p, Nat.Prime p → (p ≥ 10^2022 ∧ p < 10^2023) → p ≤ q

/-- The smallest positive integer m such that q^2 - m is divisible by 15 -/
def m : ℕ := sorry

theorem smallest_m_is_one : m = 1 := by sorry

end NUMINAMATH_CALUDE_smallest_m_is_one_l3719_371950


namespace NUMINAMATH_CALUDE_balloon_ratio_l3719_371960

theorem balloon_ratio : 
  let dan_balloons : ℝ := 29.0
  let tim_balloons : ℝ := 4.142857143
  dan_balloons / tim_balloons = 7 := by
sorry

end NUMINAMATH_CALUDE_balloon_ratio_l3719_371960


namespace NUMINAMATH_CALUDE_optimal_production_volume_l3719_371998

-- Define the profit function
def W (x : ℝ) : ℝ := -2 * x^3 + 21 * x^2

-- State the theorem
theorem optimal_production_volume (x : ℝ) (h : x > 0) :
  ∃ (max_x : ℝ), max_x = 7 ∧ 
  ∀ y, y > 0 → W y ≤ W max_x :=
sorry

end NUMINAMATH_CALUDE_optimal_production_volume_l3719_371998


namespace NUMINAMATH_CALUDE_expression_evaluation_l3719_371902

theorem expression_evaluation (a b : ℤ) (ha : a = -1) (hb : b = 1) :
  (a - 2*b) * (a^2 + 2*a*b + 4*b^2) - a * (a - 5*b) * (a + 3*b) = -21 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3719_371902


namespace NUMINAMATH_CALUDE_angle_difference_l3719_371969

theorem angle_difference (a β : ℝ) 
  (h1 : 3 * Real.sin a - Real.cos a = 0)
  (h2 : 7 * Real.sin β + Real.cos β = 0)
  (h3 : 0 < a) (h4 : a < Real.pi / 2)
  (h5 : Real.pi / 2 < β) (h6 : β < Real.pi) :
  2 * a - β = - 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_difference_l3719_371969


namespace NUMINAMATH_CALUDE_triangle_sides_not_proportional_l3719_371982

theorem triangle_sides_not_proportional (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  ¬∃ (m : ℝ), m > 0 ∧ a = m^a ∧ b = m^b ∧ c = m^c :=
sorry

end NUMINAMATH_CALUDE_triangle_sides_not_proportional_l3719_371982


namespace NUMINAMATH_CALUDE_workers_count_l3719_371912

theorem workers_count (total_work : ℕ) : ∃ (workers : ℕ),
  (workers * 65 = total_work) ∧
  ((workers + 10) * 55 = total_work) ∧
  (workers = 55) := by
sorry

end NUMINAMATH_CALUDE_workers_count_l3719_371912


namespace NUMINAMATH_CALUDE_min_value_of_y_l3719_371949

def y (x : ℝ) : ℝ :=
  |x - 1| + |x - 2| + |x - 3| + |x - 4| + |x - 5| + |x - 6| + |x - 7| + |x - 8| + |x - 9| + |x - 10|

theorem min_value_of_y :
  ∃ (x : ℝ), ∀ (z : ℝ), y z ≥ y x ∧ y x = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_y_l3719_371949


namespace NUMINAMATH_CALUDE_single_elimination_tournament_matches_l3719_371953

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  initial_players : ℕ
  matches_played : ℕ

/-- The number of matches needed to determine a champion in a single-elimination tournament -/
def matches_needed (tournament : SingleEliminationTournament) : ℕ :=
  tournament.initial_players - 1

theorem single_elimination_tournament_matches 
  (tournament : SingleEliminationTournament)
  (h : tournament.initial_players = 512) :
  matches_needed tournament = 511 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_matches_l3719_371953


namespace NUMINAMATH_CALUDE_alyssa_book_count_l3719_371900

/-- The number of books Alyssa has -/
def alyssas_books : ℕ := 36

/-- The number of books Nancy has -/
def nancys_books : ℕ := 252

theorem alyssa_book_count :
  (nancys_books = 7 * alyssas_books) → alyssas_books = 36 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_book_count_l3719_371900


namespace NUMINAMATH_CALUDE_smallest_n_cube_and_square_l3719_371975

theorem smallest_n_cube_and_square : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (a : ℕ), 4 * n = a^3) ∧ 
  (∃ (b : ℕ), 5 * n = b^2) ∧ 
  (∀ (m : ℕ), m > 0 → 
    (∃ (c : ℕ), 4 * m = c^3) → 
    (∃ (d : ℕ), 5 * m = d^2) → 
    m ≥ n) ∧
  n = 400 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_cube_and_square_l3719_371975


namespace NUMINAMATH_CALUDE_length_of_AB_l3719_371984

-- Define the triangle
def Triangle (A B C : ℝ) := True

-- Define the right angle
def is_right_angle (B : ℝ) := B = 90

-- Define the angle A
def angle_A (A : ℝ) := A = 40

-- Define the length of side BC
def side_BC (BC : ℝ) := BC = 7

-- Theorem statement
theorem length_of_AB (A B C BC : ℝ) 
  (triangle : Triangle A B C) 
  (right_angle : is_right_angle B) 
  (angle_a : angle_A A) 
  (side_bc : side_BC BC) : 
  ∃ (AB : ℝ), abs (AB - 8.3) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AB_l3719_371984


namespace NUMINAMATH_CALUDE_properties_of_negative_23_l3719_371937

theorem properties_of_negative_23 :
  let x : ℝ := -23
  (∃ y : ℝ, x + y = 0 ∧ y = 23) ∧
  (∃ z : ℝ, x * z = 1 ∧ z = -1/23) ∧
  (abs x = 23) := by
  sorry

end NUMINAMATH_CALUDE_properties_of_negative_23_l3719_371937


namespace NUMINAMATH_CALUDE_chinese_barrel_stack_l3719_371911

/-- Calculates the total number of barrels in a terraced stack --/
def totalBarrels (a b n : ℕ) : ℕ :=
  let c := a + n - 1
  let d := b + n - 1
  (n * ((2 * a + c) * b + (2 * c + a) * d + (d - b))) / 6

/-- The problem statement --/
theorem chinese_barrel_stack : totalBarrels 2 1 15 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_chinese_barrel_stack_l3719_371911


namespace NUMINAMATH_CALUDE_norm_equals_5_sqrt_5_l3719_371909

def vector : Fin 2 → ℝ
  | 0 => 3
  | 1 => 1

theorem norm_equals_5_sqrt_5 (k : ℝ) : 
  ∃ (v : Fin 2 → ℝ), v 0 = -5 ∧ v 1 = 6 ∧
  (‖(k • vector - v)‖ = 5 * Real.sqrt 5) ↔ 
  (k = (-9 + Real.sqrt 721) / 10 ∨ k = (-9 - Real.sqrt 721) / 10) :=
by sorry

end NUMINAMATH_CALUDE_norm_equals_5_sqrt_5_l3719_371909


namespace NUMINAMATH_CALUDE_power_equality_l3719_371974

theorem power_equality (m : ℕ) : 5^m = 5 * 25^5 * 125^3 → m = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3719_371974


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3719_371985

theorem first_discount_percentage (original_price final_price : ℝ) (second_discount : ℝ) :
  original_price = 26.67 →
  final_price = 15 →
  second_discount = 25 →
  ∃ first_discount : ℝ,
    first_discount = 25 ∧
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l3719_371985


namespace NUMINAMATH_CALUDE_college_student_count_l3719_371992

theorem college_student_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 160) :
  boys + girls = 416 := by
sorry

end NUMINAMATH_CALUDE_college_student_count_l3719_371992


namespace NUMINAMATH_CALUDE_max_lessons_is_216_l3719_371910

/-- Represents the teacher's wardrobe and lesson capacity. -/
structure TeacherWardrobe where
  shirts : ℕ
  pants : ℕ
  shoes : ℕ
  jackets : ℕ
  lesson_count : ℕ

/-- Calculates the number of lessons possible with the given wardrobe. -/
def calculate_lessons (w : TeacherWardrobe) : ℕ :=
  2 * w.shirts * w.pants * w.shoes

/-- Checks if the wardrobe satisfies the given conditions. -/
def satisfies_conditions (w : TeacherWardrobe) : Prop :=
  w.jackets = 2 ∧
  calculate_lessons { w with shirts := w.shirts + 1 } = w.lesson_count + 36 ∧
  calculate_lessons { w with pants := w.pants + 1 } = w.lesson_count + 72 ∧
  calculate_lessons { w with shoes := w.shoes + 1 } = w.lesson_count + 54

/-- The theorem stating the maximum number of lessons. -/
theorem max_lessons_is_216 :
  ∃ (w : TeacherWardrobe), satisfies_conditions w ∧ w.lesson_count = 216 ∧
  ∀ (w' : TeacherWardrobe), satisfies_conditions w' → w'.lesson_count ≤ 216 :=
sorry

end NUMINAMATH_CALUDE_max_lessons_is_216_l3719_371910


namespace NUMINAMATH_CALUDE_vectors_form_basis_l3719_371999

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

theorem vectors_form_basis : 
  LinearIndependent ℝ (fun i => if i = 0 then e₁ else e₂) ∧ 
  Submodule.span ℝ {e₁, e₂} = ⊤ :=
sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l3719_371999


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3719_371928

theorem imaginary_part_of_z (z : ℂ) (h : z / (2 - I) = I) : z.im = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3719_371928


namespace NUMINAMATH_CALUDE_initial_roses_count_l3719_371971

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 3

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 12

/-- The number of roses after adding flowers -/
def final_roses : ℕ := 11

/-- The number of orchids after adding flowers -/
def final_orchids : ℕ := 20

/-- The difference between orchids and roses after adding flowers -/
def orchid_rose_difference : ℕ := 9

theorem initial_roses_count :
  initial_roses = 3 ∧
  initial_orchids = 12 ∧
  final_roses = 11 ∧
  final_orchids = 20 ∧
  orchid_rose_difference = 9 ∧
  final_orchids - final_roses = orchid_rose_difference ∧
  final_orchids - initial_orchids = final_roses - initial_roses :=
by sorry

end NUMINAMATH_CALUDE_initial_roses_count_l3719_371971


namespace NUMINAMATH_CALUDE_y_value_at_x_8_l3719_371988

theorem y_value_at_x_8 (k : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y = k * x^(1/3)) →
  (∃ y : ℝ, y = 4 * Real.sqrt 3 ∧ 64^(1/3) * k = y) →
  ∃ y : ℝ, y = 2 * Real.sqrt 3 ∧ 8^(1/3) * k = y :=
by sorry

end NUMINAMATH_CALUDE_y_value_at_x_8_l3719_371988


namespace NUMINAMATH_CALUDE_unique_six_digit_number_divisibility_l3719_371965

def is_valid_digit (d : Nat) : Prop := d ≥ 1 ∧ d ≤ 6

def all_digits_unique (p q r s t u : Nat) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧
  r ≠ s ∧ r ≠ t ∧ r ≠ u ∧
  s ≠ t ∧ s ≠ u ∧
  t ≠ u

def three_digit_number (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

theorem unique_six_digit_number_divisibility (p q r s t u : Nat) :
  is_valid_digit p ∧ is_valid_digit q ∧ is_valid_digit r ∧
  is_valid_digit s ∧ is_valid_digit t ∧ is_valid_digit u ∧
  all_digits_unique p q r s t u ∧
  (three_digit_number p q r) % 4 = 0 ∧
  (three_digit_number q r s) % 6 = 0 ∧
  (three_digit_number r s t) % 3 = 0 →
  p = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_divisibility_l3719_371965


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3719_371921

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 150 → volume = 125 → ∃ (side_length : ℝ), 
    surface_area = 6 * side_length^2 ∧ volume = side_length^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3719_371921


namespace NUMINAMATH_CALUDE_spade_then_king_probability_l3719_371943

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of decks shuffled together -/
def num_decks : ℕ := 2

/-- The total number of cards after shuffling -/
def total_cards : ℕ := standard_deck_size * num_decks

/-- The number of spades in a standard deck -/
def spades_per_deck : ℕ := 13

/-- The number of kings in a standard deck -/
def kings_per_deck : ℕ := 4

/-- The probability of drawing a spade as the first card and a king as the second card -/
theorem spade_then_king_probability : 
  (spades_per_deck * num_decks) / total_cards * 
  (kings_per_deck * num_decks) / (total_cards - 1) = 103 / 5356 := by
  sorry

end NUMINAMATH_CALUDE_spade_then_king_probability_l3719_371943


namespace NUMINAMATH_CALUDE_square_of_85_l3719_371915

theorem square_of_85 : (85 : ℕ)^2 = 7225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_85_l3719_371915


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3719_371929

theorem isosceles_triangle_perimeter : ∀ x y : ℝ,
  x^2 - 9*x + 18 = 0 →
  y^2 - 9*y + 18 = 0 →
  x ≠ y →
  (x + 2*y = 15 ∨ y + 2*x = 15) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3719_371929


namespace NUMINAMATH_CALUDE_second_expression_value_l3719_371952

/-- Given that the average of (2a + 16) and x is 79, and a = 30, prove that x = 82 -/
theorem second_expression_value (a x : ℝ) : 
  ((2 * a + 16) + x) / 2 = 79 → a = 30 → x = 82 := by
  sorry

end NUMINAMATH_CALUDE_second_expression_value_l3719_371952


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3719_371904

theorem purely_imaginary_complex_number (m : ℝ) : 
  (m^2 + 3*m - 4 = 0) ∧ (m + 4 ≠ 0) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3719_371904


namespace NUMINAMATH_CALUDE_different_color_probability_l3719_371996

/-- The probability of drawing two balls of different colors from a bag containing 3 white balls and 2 black balls -/
theorem different_color_probability (total : Nat) (white : Nat) (black : Nat) 
  (h1 : total = 5) 
  (h2 : white = 3) 
  (h3 : black = 2) 
  (h4 : total = white + black) : 
  (white * black : ℚ) / (total.choose 2) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l3719_371996


namespace NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l3719_371903

/-- An isosceles, obtuse triangle with one angle 80% larger than a right angle has two smallest angles measuring 9 degrees each. -/
theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →  -- sum of angles in a triangle
  a = b →  -- isosceles condition
  c = 90 + 0.8 * 90 →  -- largest angle is 80% larger than right angle
  a = 9 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l3719_371903


namespace NUMINAMATH_CALUDE_prime_product_33_l3719_371955

theorem prime_product_33 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p * q = 33 →
  15 < p * q →
  p * q < 36 →
  8 < q →
  q < 24 →
  p = 3 := by
sorry

end NUMINAMATH_CALUDE_prime_product_33_l3719_371955


namespace NUMINAMATH_CALUDE_plate_arrangement_theorem_l3719_371901

/-- The number of ways to arrange plates around a circular table. -/
def circularArrangements (total : ℕ) (blue red green orange : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial orange) / total

/-- The number of ways to arrange plates around a circular table with one pair adjacent. -/
def circularArrangementsWithPairAdjacent (total : ℕ) (blue red green orange : ℕ) : ℕ :=
  (Nat.factorial (total - 1)) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial orange) / (total - 1)

/-- The number of ways to arrange plates around a circular table with both pairs adjacent. -/
def circularArrangementsWithBothPairsAdjacent (total : ℕ) (blue red green orange : ℕ) : ℕ :=
  (Nat.factorial (total - 2)) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial orange) / (total - 2)

theorem plate_arrangement_theorem :
  let total := 11
  let blue := 6
  let red := 2
  let green := 2
  let orange := 1
  circularArrangements total blue red green orange -
  circularArrangementsWithPairAdjacent total blue red green orange -
  circularArrangementsWithPairAdjacent total blue red green orange +
  circularArrangementsWithBothPairsAdjacent total blue red green orange = 1568 := by
  sorry

end NUMINAMATH_CALUDE_plate_arrangement_theorem_l3719_371901


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3719_371924

theorem cube_root_equation_solution :
  ∀ x : ℝ, (5 + x / 3) ^ (1/3 : ℝ) = 2 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3719_371924


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3719_371954

/-- A line that does not pass through the origin -/
structure Line where
  slope : ℝ
  intercept : ℝ
  not_through_origin : intercept ≠ 0

/-- A point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y = x^2 -/
def parabola (p : Point) : Prop :=
  p.y = p.x^2

/-- The line intersects the parabola at two points -/
def intersects_parabola (l : Line) (A B : Point) : Prop :=
  parabola A ∧ parabola B ∧
  A.y = l.slope * A.x + l.intercept ∧
  B.y = l.slope * B.x + l.intercept ∧
  A ≠ B

/-- The circle with diameter AB passes through the origin -/
def circle_through_origin (A B : Point) : Prop :=
  A.x * B.x + A.y * B.y = 0

/-- The main theorem -/
theorem line_passes_through_fixed_point (l : Line) (A B : Point)
  (h_intersects : intersects_parabola l A B)
  (h_circle : circle_through_origin A B) :
  ∃ (P : Point), P.x = 0 ∧ P.y = 1 ∧ P.y = l.slope * P.x + l.intercept :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3719_371954


namespace NUMINAMATH_CALUDE_wall_height_proof_l3719_371967

/-- Proves that the height of a wall is 600 cm given specific conditions --/
theorem wall_height_proof (brick_length brick_width brick_height : ℝ)
                          (wall_length wall_width : ℝ)
                          (num_bricks : ℕ) :
  brick_length = 25 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_length = 850 →
  wall_width = 22.5 →
  num_bricks = 6800 →
  ∃ (wall_height : ℝ),
    wall_height = 600 ∧
    num_bricks * (brick_length * brick_width * brick_height) =
    wall_length * wall_width * wall_height :=
by
  sorry

#check wall_height_proof

end NUMINAMATH_CALUDE_wall_height_proof_l3719_371967


namespace NUMINAMATH_CALUDE_four_girls_wins_l3719_371978

theorem four_girls_wins (a b c d : ℕ) : 
  a + b = 8 ∧ 
  a + c = 10 ∧ 
  b + c = 12 ∧ 
  a + d = 12 ∧ 
  b + d = 14 ∧ 
  c + d = 16 → 
  ({a, b, c, d} : Finset ℕ) = {3, 5, 7, 9} := by
sorry

end NUMINAMATH_CALUDE_four_girls_wins_l3719_371978


namespace NUMINAMATH_CALUDE_board_cut_ratio_l3719_371930

/-- Given a board of length 69 inches cut into two pieces, where the shorter piece is 23 inches long,
    the ratio of the longer piece to the shorter piece is 2:1. -/
theorem board_cut_ratio : 
  ∀ (short_piece long_piece : ℝ),
  short_piece = 23 →
  short_piece + long_piece = 69 →
  long_piece / short_piece = 2 := by
sorry

end NUMINAMATH_CALUDE_board_cut_ratio_l3719_371930


namespace NUMINAMATH_CALUDE_simplify_expression_l3719_371922

theorem simplify_expression : 18 * (8 / 12) * (1 / 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3719_371922


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3719_371997

theorem quadratic_factorization (a b : ℤ) :
  (∀ x, 24 * x^2 - 98 * x - 168 = (6 * x + a) * (4 * x + b)) →
  a + 2 * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3719_371997


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l3719_371981

theorem sum_of_fifth_powers (a b x y : ℝ) 
  (eq1 : a*x + b*y = 3)
  (eq2 : a*x^2 + b*y^2 = 7)
  (eq3 : a*x^3 + b*y^3 = 6)
  (eq4 : a*x^4 + b*y^4 = 42) :
  a*x^5 + b*y^5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l3719_371981


namespace NUMINAMATH_CALUDE_transformation_symmetry_l3719_371945

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the transformation function
def transform (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

-- Define symmetry with respect to x-axis
def symmetricToXAxis (p1 p2 : Point2D) : Prop :=
  p1.x = p2.x ∧ p1.y = -p2.y

-- Theorem statement
theorem transformation_symmetry (p : Point2D) :
  symmetricToXAxis p (transform p) := by
  sorry


end NUMINAMATH_CALUDE_transformation_symmetry_l3719_371945


namespace NUMINAMATH_CALUDE_min_area_rectangle_l3719_371976

theorem min_area_rectangle (l w : ℕ) : 
  l > 0 → w > 0 → 2 * (l + w) = 150 → l * w ≥ 74 := by
  sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l3719_371976


namespace NUMINAMATH_CALUDE_chord_length_in_unit_circle_l3719_371913

theorem chord_length_in_unit_circle (chord1 chord2 chord3 : Real) : 
  -- Unit circle condition
  ∀ (r : Real), r = 1 →
  -- Three distinct diameters
  ∃ (α θ : Real), α ≠ θ ∧ α + θ + (180 - α - θ) = 180 →
  -- One chord has length √2
  chord1 = Real.sqrt 2 →
  -- The other two chords have equal lengths
  chord2 = chord3 →
  -- Length of chord2 and chord3
  chord2 = Real.sqrt (2 - Real.sqrt 2) := by
sorry


end NUMINAMATH_CALUDE_chord_length_in_unit_circle_l3719_371913


namespace NUMINAMATH_CALUDE_intersection_points_l3719_371991

-- Define g as a function from real numbers to real numbers
variable (g : ℝ → ℝ)

-- Define the property that g is invertible
def IsInvertible (g : ℝ → ℝ) : Prop :=
  ∃ h : ℝ → ℝ, (∀ x, h (g x) = x) ∧ (∀ y, g (h y) = y)

-- Theorem statement
theorem intersection_points (h : IsInvertible g) :
  (∃! n : Nat, ∃ s : Finset ℝ, s.card = n ∧
    (∀ x : ℝ, x ∈ s ↔ g (x^3) = g (x^6))) ∧
  (∃ s : Finset ℝ, s.card = 3 ∧
    (∀ x : ℝ, x ∈ s ↔ g (x^3) = g (x^6))) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_l3719_371991


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l3719_371934

theorem halfway_between_fractions :
  (3 / 4 + 5 / 6) / 2 = 19 / 24 := by sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l3719_371934


namespace NUMINAMATH_CALUDE_storks_on_fence_l3719_371926

/-- The number of storks on a fence, given the initial number of birds,
    the number of birds that join, and the final difference between birds and storks. -/
def number_of_storks (initial_birds : ℕ) (joining_birds : ℕ) (final_difference : ℕ) : ℕ :=
  initial_birds + joining_birds - final_difference

/-- Theorem stating that the number of storks is 4 under the given conditions. -/
theorem storks_on_fence : number_of_storks 3 2 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_storks_on_fence_l3719_371926


namespace NUMINAMATH_CALUDE_integer_ratio_problem_l3719_371959

theorem integer_ratio_problem (a b c : ℕ) : 
  a < b → b < c → 
  a = 0 → b ≠ a + 1 → 
  (a + b + c : ℚ) / 3 = 4 * b → 
  c / b = 11 := by
  sorry

end NUMINAMATH_CALUDE_integer_ratio_problem_l3719_371959


namespace NUMINAMATH_CALUDE_waiter_customers_l3719_371948

/-- Calculates the total number of customers for a waiter given the number of tables and customers per table. -/
def total_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) : ℕ :=
  num_tables * (women_per_table + men_per_table)

/-- Proves that a waiter with 9 tables, each having 7 women and 3 men, has 90 customers in total. -/
theorem waiter_customers : total_customers 9 7 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l3719_371948


namespace NUMINAMATH_CALUDE_x_value_from_fraction_equality_l3719_371932

theorem x_value_from_fraction_equality (x y : ℝ) :
  x / (x - 1) = (y^2 + 2*y + 3) / (y^2 + 2*y - 2) →
  x = (y^2 + 2*y + 3) / 5 := by
sorry

end NUMINAMATH_CALUDE_x_value_from_fraction_equality_l3719_371932


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3719_371938

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3719_371938


namespace NUMINAMATH_CALUDE_outfits_count_l3719_371919

/-- The number of shirts available. -/
def num_shirts : ℕ := 5

/-- The number of pairs of pants available. -/
def num_pants : ℕ := 3

/-- The number of ties available. -/
def num_ties : ℕ := 2

/-- The total number of possible outfits. -/
def total_outfits : ℕ := num_shirts * num_pants * num_ties

/-- Theorem stating that the total number of possible outfits is 30. -/
theorem outfits_count : total_outfits = 30 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l3719_371919


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3719_371977

/-- Proves the existence and uniqueness of the intersection point of two lines, if it exists -/
theorem intersection_of_lines (a b c d e f : ℝ) (h1 : a ≠ 0 ∨ b ≠ 0) (h2 : c ≠ 0 ∨ d ≠ 0) 
  (h3 : a * d ≠ b * c) : 
  ∃! p : ℝ × ℝ, a * p.1 + b * p.2 + e = 0 ∧ c * p.1 + d * p.2 + f = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3719_371977


namespace NUMINAMATH_CALUDE_division_reduction_l3719_371993

theorem division_reduction (x : ℝ) (h : x > 0) : 54 / x = 54 - 36 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_reduction_l3719_371993


namespace NUMINAMATH_CALUDE_simplify_expression_l3719_371963

theorem simplify_expression (x : ℝ) (h : x ≥ 0) : 
  (1/2 * x^(1/2))^4 = 1/16 * x^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3719_371963


namespace NUMINAMATH_CALUDE_exercise_book_count_l3719_371941

theorem exercise_book_count (pencil_count : ℕ) (pencil_ratio : ℕ) (book_ratio : ℕ) :
  pencil_count = 120 →
  pencil_ratio = 10 →
  book_ratio = 3 →
  (pencil_count * book_ratio) / pencil_ratio = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_exercise_book_count_l3719_371941


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3719_371927

theorem regular_polygon_sides (n : ℕ) (h : n > 0) :
  (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3719_371927


namespace NUMINAMATH_CALUDE_sin_graph_transformation_l3719_371947

theorem sin_graph_transformation (x : ℝ) :
  let f (x : ℝ) := 2 * Real.sin x
  let g (x : ℝ) := 2 * Real.sin (x / 3 + π / 6)
  let h (x : ℝ) := f (x + π / 6)
  g x = h (x / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_graph_transformation_l3719_371947


namespace NUMINAMATH_CALUDE_water_bottles_used_second_game_l3719_371925

theorem water_bottles_used_second_game 
  (initial_cases : ℕ)
  (bottles_per_case : ℕ)
  (bottles_used_first_game : ℕ)
  (bottles_remaining_after_second_game : ℕ)
  (h1 : initial_cases = 10)
  (h2 : bottles_per_case = 20)
  (h3 : bottles_used_first_game = 70)
  (h4 : bottles_remaining_after_second_game = 20) :
  initial_cases * bottles_per_case - bottles_used_first_game - bottles_remaining_after_second_game = 110 :=
by sorry

end NUMINAMATH_CALUDE_water_bottles_used_second_game_l3719_371925


namespace NUMINAMATH_CALUDE_min_clients_theorem_exists_solution_with_101_min_clients_is_101_l3719_371987

/-- Represents a repunit number with n ones -/
def repunit (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- The property that needs to be satisfied for the group of clients -/
def satisfies_property (m k : ℕ) : Prop :=
  ∃ n : ℕ, n > k ∧ k > 1 ∧ repunit n = repunit k * m

/-- The main theorem stating the minimum number of clients -/
theorem min_clients_theorem :
  ∀ m : ℕ, m > 1 → (satisfies_property m 2) → m ≥ 101 :=
by sorry

/-- The existence theorem proving there is a solution with 101 clients -/
theorem exists_solution_with_101 :
  satisfies_property 101 2 :=
by sorry

/-- The final theorem proving 101 is the minimum number of clients -/
theorem min_clients_is_101 :
  ∀ m : ℕ, m > 1 → satisfies_property m 2 → m ≥ 101 ∧ satisfies_property 101 2 :=
by sorry

end NUMINAMATH_CALUDE_min_clients_theorem_exists_solution_with_101_min_clients_is_101_l3719_371987


namespace NUMINAMATH_CALUDE_hypotenuse_length_l3719_371914

/-- A point on the parabola y = -x^2 --/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = -x^2

/-- Triangle ABO with A and B on the parabola y = -x^2 and ∠AOB = 45° --/
structure TriangleABO where
  A : ParabolaPoint
  B : ParabolaPoint
  angle_AOB : Real.pi / 4 = Real.arctan (A.y / A.x) + Real.arctan (B.y / B.x)

/-- The length of the hypotenuse of triangle ABO is 2 --/
theorem hypotenuse_length (t : TriangleABO) : 
  Real.sqrt ((t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l3719_371914


namespace NUMINAMATH_CALUDE_prob_two_green_apples_l3719_371980

/-- The probability of selecting 2 green apples from a set of 7 apples, where 3 are green -/
theorem prob_two_green_apples (total : ℕ) (green : ℕ) (choose : ℕ) 
  (h_total : total = 7) 
  (h_green : green = 3) 
  (h_choose : choose = 2) :
  (Nat.choose green choose : ℚ) / (Nat.choose total choose : ℚ) = 1 / 7 := by
  sorry

#check prob_two_green_apples

end NUMINAMATH_CALUDE_prob_two_green_apples_l3719_371980
