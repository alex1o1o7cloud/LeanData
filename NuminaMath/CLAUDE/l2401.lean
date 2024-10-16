import Mathlib

namespace NUMINAMATH_CALUDE_count_threes_to_1000_l2401_240101

/-- Count of digit 3 appearances when listing integers from 1 to n -/
def count_threes (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the count of digit 3 appearances from 1 to 1000 is 300 -/
theorem count_threes_to_1000 : count_threes 1000 = 300 := by
  sorry

end NUMINAMATH_CALUDE_count_threes_to_1000_l2401_240101


namespace NUMINAMATH_CALUDE_largest_divisor_of_Q_l2401_240197

/-- Definition of Q as the product of two consecutive even numbers and their preceding odd number -/
def Q (n : ℕ) : ℕ := (2*n - 1) * (2*n) * (2*n + 2)

/-- Theorem stating that 8 is the largest integer that divides all Q -/
theorem largest_divisor_of_Q :
  ∀ k : ℕ, (∀ n : ℕ, k ∣ Q n) → k ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_Q_l2401_240197


namespace NUMINAMATH_CALUDE_symmetry_xoz_plane_l2401_240182

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the xOz plane
def xOzPlane : Set Point3D := {p : Point3D | p.y = 0}

-- Define symmetry with respect to the xOz plane
def symmetricPointXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetry_xoz_plane :
  let P := Point3D.mk 3 1 5
  let Q := Point3D.mk 3 (-1) 5
  symmetricPointXOZ P = Q := by sorry

end NUMINAMATH_CALUDE_symmetry_xoz_plane_l2401_240182


namespace NUMINAMATH_CALUDE_george_oranges_l2401_240135

theorem george_oranges (george_oranges : ℕ) (george_apples : ℕ) (amelia_oranges : ℕ) (amelia_apples : ℕ) : 
  george_apples = amelia_apples + 5 →
  amelia_oranges = george_oranges - 18 →
  amelia_apples = 15 →
  george_oranges + george_apples + amelia_oranges + amelia_apples = 107 →
  george_oranges = 45 := by
sorry

end NUMINAMATH_CALUDE_george_oranges_l2401_240135


namespace NUMINAMATH_CALUDE_factorization_2x_cubed_minus_8x_l2401_240173

theorem factorization_2x_cubed_minus_8x (x : ℝ) : 2*x^3 - 8*x = 2*x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2x_cubed_minus_8x_l2401_240173


namespace NUMINAMATH_CALUDE_prime_representation_mod_24_l2401_240130

theorem prime_representation_mod_24 (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  (∃ x y : ℤ, (p : ℤ) = 2 * x^2 + 3 * y^2) ↔ (p % 24 = 5 ∨ p % 24 = 11) :=
by sorry

end NUMINAMATH_CALUDE_prime_representation_mod_24_l2401_240130


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l2401_240131

theorem normal_distribution_std_dev (μ : ℝ) (x : ℝ) (σ : ℝ) 
  (h1 : μ = 14.5)
  (h2 : x = 11.1)
  (h3 : x = μ - 2 * σ) :
  σ = 1.7 := by
sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l2401_240131


namespace NUMINAMATH_CALUDE_max_correct_answers_l2401_240109

theorem max_correct_answers (total_questions : ℕ) (score : ℤ) : 
  total_questions = 25 → score = 65 → 
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    4 * correct - incorrect = score ∧
    correct ≤ 18 ∧
    ∀ c i u : ℕ, 
      c + i + u = total_questions → 
      4 * c - i = score → 
      c ≤ correct :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l2401_240109


namespace NUMINAMATH_CALUDE_highest_power_of_three_dividing_M_l2401_240148

def M : ℕ := sorry

theorem highest_power_of_three_dividing_M :
  ∃ (j : ℕ), (3^j ∣ M) ∧ ¬(3^(j+1) ∣ M) ∧ j = 1 := by sorry

end NUMINAMATH_CALUDE_highest_power_of_three_dividing_M_l2401_240148


namespace NUMINAMATH_CALUDE_total_income_calculation_l2401_240164

def original_cupcake_price : ℚ := 3
def original_cookie_price : ℚ := 2
def cupcake_discount : ℚ := 0.3
def cookie_discount : ℚ := 0.45
def cupcakes_sold : ℕ := 25
def cookies_sold : ℕ := 18

theorem total_income_calculation :
  let new_cupcake_price := original_cupcake_price * (1 - cupcake_discount)
  let new_cookie_price := original_cookie_price * (1 - cookie_discount)
  let total_income := (new_cupcake_price * cupcakes_sold) + (new_cookie_price * cookies_sold)
  total_income = 72.3 := by sorry

end NUMINAMATH_CALUDE_total_income_calculation_l2401_240164


namespace NUMINAMATH_CALUDE_gpa_ratio_is_one_third_l2401_240113

/-- Represents a class with two groups of students with different GPAs -/
structure ClassGPA where
  totalStudents : ℕ
  studentsGPA30 : ℕ
  gpa30 : ℝ := 30
  gpa33 : ℝ := 33
  overallGPA : ℝ := 32

/-- The ratio of students with GPA 30 to the total number of students is 1/3 -/
theorem gpa_ratio_is_one_third (c : ClassGPA) 
  (h1 : c.studentsGPA30 ≤ c.totalStudents)
  (h2 : c.totalStudents > 0)
  (h3 : c.gpa30 * c.studentsGPA30 + c.gpa33 * (c.totalStudents - c.studentsGPA30) = c.overallGPA * c.totalStudents) :
  c.studentsGPA30 / c.totalStudents = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gpa_ratio_is_one_third_l2401_240113


namespace NUMINAMATH_CALUDE_integral_f_equals_one_plus_pi_over_four_l2401_240117

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then Real.sqrt (1 - x^2)
  else if -1 ≤ x ∧ x ≤ 0 then x + 1
  else 0  -- This case is added to make the function total

-- State the theorem
theorem integral_f_equals_one_plus_pi_over_four :
  ∫ x in (-1)..1, f x = (1 + Real.pi) / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_integral_f_equals_one_plus_pi_over_four_l2401_240117


namespace NUMINAMATH_CALUDE_kates_hair_length_l2401_240112

/-- Given information about hair lengths of Kate, Emily, and Logan, prove Kate's hair length -/
theorem kates_hair_length (logan_length emily_length kate_length : ℝ) : 
  logan_length = 20 →
  emily_length = logan_length + 6 →
  kate_length = emily_length / 2 →
  kate_length = 13 := by
  sorry

end NUMINAMATH_CALUDE_kates_hair_length_l2401_240112


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2401_240128

/-- A linear function y = ax + b where y increases as x increases and ab < 0 -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  increasing : a > 0
  product_negative : a * b < 0

/-- The point P(a,b) -/
def point (f : LinearFunction) : ℝ × ℝ := (f.a, f.b)

/-- A point (x,y) lies in the fourth quadrant if x > 0 and y < 0 -/
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant (f : LinearFunction) :
  in_fourth_quadrant (point f) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2401_240128


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l2401_240160

theorem triangle_angle_inequality (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) : 
  π * (1/A + 1/B + 1/C) ≥ (Real.sin A + Real.sin B + Real.sin C) * 
    (1/Real.sin A + 1/Real.sin B + 1/Real.sin C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l2401_240160


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l2401_240124

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 1230000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 1.23
    exponent := 6
    valid := by sorry }

/-- Theorem stating that the scientific notation representation is correct -/
theorem scientific_notation_correct :
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l2401_240124


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2401_240159

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- Angles are positive
  A + B + C = π ∧ -- Sum of angles in a triangle
  0 < a ∧ 0 < b ∧ 0 < c ∧ -- Sides are positive
  b * Real.sin (2 * C) = c * Real.sin B ∧ -- Given condition
  Real.sin (B - π / 3) = 3 / 5 -- Given condition
  →
  C = π / 3 ∧ 
  Real.sin A = (4 * Real.sqrt 3 - 3) / 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2401_240159


namespace NUMINAMATH_CALUDE_pizza_problem_l2401_240195

theorem pizza_problem : ∃! (m d : ℕ), 
  m > 0 ∧ d > 0 ∧ 
  7 * m + 2 * d > 36 ∧
  8 * m + 4 * d < 48 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l2401_240195


namespace NUMINAMATH_CALUDE_game_score_invariant_final_score_difference_l2401_240115

def game_score (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem game_score_invariant (n : ℕ) (h : n ≥ 2) :
  ∀ (moves : List (ℕ × ℕ × ℕ)),
    moves.all (λ (a, b, c) => a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 1 ∧ b + c = a) →
    moves.foldl (λ acc (a, b, c) => acc + b * c) 0 = game_score n :=
  sorry

theorem final_score_difference (n : ℕ) (h : n ≥ 2) :
  let M := game_score n
  let m := game_score n
  M - m = 0 :=
  sorry

end NUMINAMATH_CALUDE_game_score_invariant_final_score_difference_l2401_240115


namespace NUMINAMATH_CALUDE_xyz_value_l2401_240192

theorem xyz_value (x y z : ℝ) : 
  4 * (Real.sqrt x + Real.sqrt (y - 1) + Real.sqrt (z - 2)) = x + y + z + 9 →
  x * y * z = 120 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2401_240192


namespace NUMINAMATH_CALUDE_base_conversion_1729_l2401_240188

theorem base_conversion_1729 :
  2 * (5 ^ 4) + 3 * (5 ^ 3) + 4 * (5 ^ 2) + 0 * (5 ^ 1) + 4 * (5 ^ 0) = 1729 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1729_l2401_240188


namespace NUMINAMATH_CALUDE_repunit_243_divisible_by_243_l2401_240102

/-- The number formed by n consecutive ones -/
def repunit (n : ℕ) : ℕ := (10^n - 1) / 9

/-- Theorem: The repunit of 243 is divisible by 243 -/
theorem repunit_243_divisible_by_243 : 243 ∣ repunit 243 := by
  sorry

end NUMINAMATH_CALUDE_repunit_243_divisible_by_243_l2401_240102


namespace NUMINAMATH_CALUDE_bella_win_probability_l2401_240152

theorem bella_win_probability (lose_prob : ℚ) (no_tie : Bool) : lose_prob = 5/11 ∧ no_tie = true → 1 - lose_prob = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_bella_win_probability_l2401_240152


namespace NUMINAMATH_CALUDE_count_385_consecutive_sums_l2401_240125

/-- Represents a sequence of consecutive positive integers -/
structure ConsecutiveSequence where
  start : ℕ
  length : ℕ
  length_ge_two : length ≥ 2

/-- The sum of a consecutive sequence -/
def sum_consecutive_sequence (seq : ConsecutiveSequence) : ℕ :=
  seq.length * (2 * seq.start + seq.length - 1) / 2

/-- Predicate for a valid sequence summing to 385 -/
def is_valid_sequence (seq : ConsecutiveSequence) : Prop :=
  sum_consecutive_sequence seq = 385

/-- The main theorem statement -/
theorem count_385_consecutive_sums :
  (∃ (seqs : Finset ConsecutiveSequence), 
    (∀ seq ∈ seqs, is_valid_sequence seq) ∧ 
    (∀ seq, is_valid_sequence seq → seq ∈ seqs) ∧
    seqs.card = 9) := by
  sorry

end NUMINAMATH_CALUDE_count_385_consecutive_sums_l2401_240125


namespace NUMINAMATH_CALUDE_tens_digit_of_2013_squared_minus_2013_l2401_240153

theorem tens_digit_of_2013_squared_minus_2013 : (2013^2 - 2013) % 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_2013_squared_minus_2013_l2401_240153


namespace NUMINAMATH_CALUDE_banks_investment_count_l2401_240170

/-- The number of investments Mr. Banks has -/
def banks_investments : ℕ := sorry

/-- The revenue Mr. Banks receives from each investment -/
def banks_revenue_per_investment : ℕ := 500

/-- The number of investment streams Ms. Elizabeth has -/
def elizabeth_investments : ℕ := 5

/-- The revenue Ms. Elizabeth receives from each investment stream -/
def elizabeth_revenue_per_investment : ℕ := 900

/-- The difference between Ms. Elizabeth's total revenue and Mr. Banks' total revenue -/
def revenue_difference : ℕ := 500

theorem banks_investment_count :
  banks_investments = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_banks_investment_count_l2401_240170


namespace NUMINAMATH_CALUDE_inequality_interval_length_l2401_240139

/-- Given an inequality a ≤ 3x + 4 ≤ b, if the length of the interval of solutions is 8, then b - a = 24 -/
theorem inequality_interval_length (a b : ℝ) : 
  (∃ (l : ℝ), l = 8 ∧ l = (b - 4) / 3 - (a - 4) / 3) → b - a = 24 :=
by sorry

end NUMINAMATH_CALUDE_inequality_interval_length_l2401_240139


namespace NUMINAMATH_CALUDE_skewReflectionAndShrinkIsCorrectTransformation_l2401_240118

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A rigid transformation in 2D space -/
structure RigidTransformation where
  transform : Point2D → Point2D

/-- Skew-reflection across y=x followed by a vertical shrink by a factor of -1 -/
def skewReflectionAndShrink : RigidTransformation :=
  { transform := λ p => Point2D.mk p.y (-p.x) }

theorem skewReflectionAndShrinkIsCorrectTransformation :
  let C := Point2D.mk 3 (-2)
  let D := Point2D.mk 4 (-3)
  let C' := Point2D.mk 1 2
  let D' := Point2D.mk (-2) 3
  (skewReflectionAndShrink.transform C = C') ∧
  (skewReflectionAndShrink.transform D = D') := by
  sorry


end NUMINAMATH_CALUDE_skewReflectionAndShrinkIsCorrectTransformation_l2401_240118


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2401_240158

theorem arithmetic_expression_equality : 5 + 16 / 4 - 3^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2401_240158


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l2401_240165

theorem smallest_constant_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ p : ℝ, ∀ a b : ℝ, 0 < a → 0 < b →
    Real.sqrt (a * b) - (2 * a * b) / (a + b) ≤ p * ((a + b) / 2 - Real.sqrt (a * b))) ∧
  (∀ p : ℝ, (∀ a b : ℝ, 0 < a → 0 < b →
    Real.sqrt (a * b) - (2 * a * b) / (a + b) ≤ p * ((a + b) / 2 - Real.sqrt (a * b))) →
    1 ≤ p) ∧
  (∀ a b : ℝ, 0 < a → 0 < b →
    Real.sqrt (a * b) - (2 * a * b) / (a + b) ≤ (a + b) / 2 - Real.sqrt (a * b)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l2401_240165


namespace NUMINAMATH_CALUDE_train_length_calculation_l2401_240155

-- Define the given values
def train_speed : Real := 63  -- km/hr
def man_speed : Real := 3     -- km/hr
def crossing_time : Real := 29.997600191984642  -- seconds

-- Define the theorem
theorem train_length_calculation :
  let relative_speed := (train_speed - man_speed) * 1000 / 3600  -- Convert to m/s
  let train_length := relative_speed * crossing_time
  ∃ ε > 0, abs (train_length - 500) < ε :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2401_240155


namespace NUMINAMATH_CALUDE_percentage_problem_l2401_240168

theorem percentage_problem (P : ℝ) : 
  (P * 100 + 60 = 100) → P = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2401_240168


namespace NUMINAMATH_CALUDE_books_sold_thursday_l2401_240171

/-- Calculates the number of books sold on Thursday given the initial stock,
    sales on other days, and the percentage of books not sold. -/
theorem books_sold_thursday
  (initial_stock : ℕ)
  (sold_monday : ℕ)
  (sold_tuesday : ℕ)
  (sold_wednesday : ℕ)
  (sold_friday : ℕ)
  (percent_not_sold : ℚ)
  (h1 : initial_stock = 1100)
  (h2 : sold_monday = 75)
  (h3 : sold_tuesday = 50)
  (h4 : sold_wednesday = 64)
  (h5 : sold_friday = 135)
  (h6 : percent_not_sold = 63.45)
  : ∃ (sold_thursday : ℕ), sold_thursday = 78 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_thursday_l2401_240171


namespace NUMINAMATH_CALUDE_travel_distance_proof_l2401_240111

theorem travel_distance_proof (total_distance : ℝ) (bus_distance : ℝ) : 
  total_distance = 1800 →
  bus_distance = 720 →
  (1/3 : ℝ) * total_distance + (2/3 : ℝ) * bus_distance + bus_distance = total_distance :=
by
  sorry

end NUMINAMATH_CALUDE_travel_distance_proof_l2401_240111


namespace NUMINAMATH_CALUDE_inverse_sum_equals_six_l2401_240167

-- Define the function f
def f (x : ℝ) : ℝ := x * |x|^2

-- State the theorem
theorem inverse_sum_equals_six :
  ∃ (a b : ℝ), f a = 8 ∧ f b = -64 ∧ a + b = 6 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_six_l2401_240167


namespace NUMINAMATH_CALUDE_deer_meat_content_deer_meat_content_is_200_l2401_240122

/-- Proves that each deer contains 200 pounds of meat given the hunting conditions -/
theorem deer_meat_content (hunting_wolves : ℕ) (additional_wolves : ℕ) 
  (meat_per_wolf_per_day : ℕ) (hunting_days : ℕ) (deer_per_hunting_wolf : ℕ) : ℕ :=
  let total_wolves := hunting_wolves + additional_wolves
  let total_meat_needed := total_wolves * meat_per_wolf_per_day * hunting_days
  let total_deer := hunting_wolves * deer_per_hunting_wolf
  total_meat_needed / total_deer

#check deer_meat_content 4 16 8 5 1 = 200

/-- Theorem stating that under the given conditions, each deer contains 200 pounds of meat -/
theorem deer_meat_content_is_200 : 
  deer_meat_content 4 16 8 5 1 = 200 := by
  sorry

end NUMINAMATH_CALUDE_deer_meat_content_deer_meat_content_is_200_l2401_240122


namespace NUMINAMATH_CALUDE_wage_difference_proof_l2401_240174

/-- Proves that the difference between hourly wages of two candidates is $5 
    given specific conditions about their pay and work hours. -/
theorem wage_difference_proof (total_pay hours_p hours_q wage_p wage_q : ℝ) 
  (h1 : total_pay = 300)
  (h2 : wage_p = 1.5 * wage_q)
  (h3 : hours_q = hours_p + 10)
  (h4 : wage_p * hours_p = total_pay)
  (h5 : wage_q * hours_q = total_pay) :
  wage_p - wage_q = 5 := by
sorry

end NUMINAMATH_CALUDE_wage_difference_proof_l2401_240174


namespace NUMINAMATH_CALUDE_seating_probability_l2401_240157

/-- The number of people seated at the round table -/
def total_people : ℕ := 12

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 4

/-- The number of biology majors -/
def biology_majors : ℕ := 3

/-- The probability of the desired seating arrangement -/
def desired_probability : ℚ := 18/175

theorem seating_probability :
  let total_arrangements := (total_people - 1).factorial
  let math_block_arrangements := total_people * (math_majors - 1).factorial
  let physics_arrangements := physics_majors.factorial
  let biology_arrangements := (physics_majors + 1).choose biology_majors * biology_majors.factorial
  let favorable_arrangements := math_block_arrangements * physics_arrangements * biology_arrangements
  (favorable_arrangements : ℚ) / total_arrangements = desired_probability := by
  sorry

end NUMINAMATH_CALUDE_seating_probability_l2401_240157


namespace NUMINAMATH_CALUDE_solution_part_i_solution_part_ii_l2401_240180

-- Define the function f
def f (x : ℝ) := |2*x - 1| + |x + 1|

-- Theorem for part I
theorem solution_part_i : 
  {x : ℝ | f x ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 2/3} :=
sorry

-- Theorem for part II
theorem solution_part_ii :
  (∀ x : ℝ, ∃ a ∈ Set.Icc (-2) 1, f x ≥ f a + m) →
  m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_solution_part_i_solution_part_ii_l2401_240180


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2401_240169

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and right focus F at (c, 0),
    if a line perpendicular to y = -bx/a passes through F and intersects the left branch of the hyperbola
    at point B such that vector FB = 2 * vector FA (where A is the foot of the perpendicular),
    then the eccentricity of the hyperbola is √5. -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let F : ℝ × ℝ := (c, 0)
  let perpendicular_line := {(x, y) : ℝ × ℝ | y = a / b * (x - c)}
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let A := (a^2 / c, a * b / c)
  ∃ B : ℝ × ℝ, B.1 < 0 ∧ B ∈ hyperbola ∧ B ∈ perpendicular_line ∧
    (B.1 - F.1, B.2 - F.2) = (2 * (A.1 - F.1), 2 * (A.2 - F.2)) →
  c^2 / a^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2401_240169


namespace NUMINAMATH_CALUDE_equation_solutions_l2401_240156

theorem equation_solutions :
  ∀ x : ℝ, 
    Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6 ↔ 
    x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2401_240156


namespace NUMINAMATH_CALUDE_factors_of_given_number_l2401_240110

/-- The number of distinct natural-number factors of 4^5 · 5^3 · 7^2 -/
def num_factors : ℕ := 132

/-- The given number -/
def given_number : ℕ := 4^5 * 5^3 * 7^2

/-- A function that counts the number of distinct natural-number factors of a given natural number -/
def count_factors (n : ℕ) : ℕ := sorry

theorem factors_of_given_number :
  count_factors given_number = num_factors := by sorry

end NUMINAMATH_CALUDE_factors_of_given_number_l2401_240110


namespace NUMINAMATH_CALUDE_divisibility_prime_factorization_l2401_240187

theorem divisibility_prime_factorization (a b : ℕ) : 
  (a ∣ b) ↔ (∀ p : ℕ, ∀ k : ℕ, Prime p → (p^k ∣ a) → (p^k ∣ b)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_prime_factorization_l2401_240187


namespace NUMINAMATH_CALUDE_nicky_dmv_wait_l2401_240136

/-- The time Nicky spent waiting to take a number, in minutes. -/
def initial_wait : ℕ := 20

/-- The time Nicky spent waiting for his number to be called, in minutes. -/
def number_wait : ℕ := 4 * initial_wait + 14

/-- The total time Nicky spent waiting at the DMV, in minutes. -/
def total_wait : ℕ := initial_wait + number_wait

theorem nicky_dmv_wait : total_wait = 114 := by
  sorry

end NUMINAMATH_CALUDE_nicky_dmv_wait_l2401_240136


namespace NUMINAMATH_CALUDE_coconut_price_l2401_240120

/-- The price of a coconut given the yield per tree, total money needed, and number of trees to harvest. -/
theorem coconut_price
  (yield_per_tree : ℕ)  -- Number of coconuts per tree
  (total_money : ℕ)     -- Total money needed in dollars
  (trees_to_harvest : ℕ) -- Number of trees to harvest
  (h1 : yield_per_tree = 5)
  (h2 : total_money = 90)
  (h3 : trees_to_harvest = 6) :
  total_money / (yield_per_tree * trees_to_harvest) = 3 :=
by sorry


end NUMINAMATH_CALUDE_coconut_price_l2401_240120


namespace NUMINAMATH_CALUDE_tan_two_theta_minus_pi_over_six_l2401_240147

theorem tan_two_theta_minus_pi_over_six (θ : Real) 
  (h : 4 * Real.cos (θ + π/3) * Real.cos (θ - π/6) = Real.sin (2*θ)) : 
  Real.tan (2*θ - π/6) = Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_theta_minus_pi_over_six_l2401_240147


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2401_240137

theorem polynomial_divisibility (m n : ℤ) :
  (∀ (x y : ℤ), (107 ∣ (x^3 + m*x + n) - (y^3 + m*y + n)) → (107 ∣ (x - y))) →
  (107 ∣ m) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2401_240137


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_equation_l2401_240181

theorem equal_roots_quadratic_equation :
  ∃! p : ℝ, ∀ x : ℝ, (x^2 - p*x + p^2 = 0 → (∃! y : ℝ, y^2 - p*y + p^2 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_equation_l2401_240181


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2401_240194

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2401_240194


namespace NUMINAMATH_CALUDE_inequality_proof_l2401_240176

theorem inequality_proof (a₁ a₂ a₃ : ℝ) 
  (h₁ : 0 ≤ a₁) (h₂ : 0 ≤ a₂) (h₃ : 0 ≤ a₃) 
  (h_sum : a₁ + a₂ + a₃ = 1) : 
  a₁ * Real.sqrt a₂ + a₂ * Real.sqrt a₃ + a₃ * Real.sqrt a₁ ≤ 1 / Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2401_240176


namespace NUMINAMATH_CALUDE_train_length_proof_l2401_240149

/-- The length of a train in meters -/
def train_length : ℝ := 1200

/-- The time in seconds it takes for the train to cross a tree -/
def tree_crossing_time : ℝ := 120

/-- The time in seconds it takes for the train to pass a platform -/
def platform_passing_time : ℝ := 150

/-- The length of the platform in meters -/
def platform_length : ℝ := 300

theorem train_length_proof :
  (train_length / tree_crossing_time = (train_length + platform_length) / platform_passing_time) →
  train_length = 1200 := by
sorry

end NUMINAMATH_CALUDE_train_length_proof_l2401_240149


namespace NUMINAMATH_CALUDE_ellipse_properties_l2401_240145

def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * a = 4 * Real.sqrt 3) 
  (h4 : (2^2 / a^2) + ((Real.sqrt 2)^2 / b^2) = 1) 
  (h5 : ∃ (A B : ℝ × ℝ), A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧ 
    (A.1 + B.1) / 2 = -8/5 ∧ (A.2 + B.2) / 2 = 2/5) :
  (a^2 = 12 ∧ b^2 = 3) ∧ 
  (∀ (A B : ℝ × ℝ), A ∈ Ellipse a b → B ∈ Ellipse a b → 
    (A.1 + B.1) / 2 = -8/5 → (A.2 + B.2) / 2 = 2/5 → 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 22 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2401_240145


namespace NUMINAMATH_CALUDE_prob_less_than_five_and_even_is_one_third_l2401_240199

/-- The probability of rolling a number less than 5 on a six-sided die -/
def prob_less_than_five : ℚ := 4 / 6

/-- The probability of rolling an even number on a six-sided die -/
def prob_even : ℚ := 3 / 6

/-- The probability of rolling a number less than 5 on the first die
    and an even number on the second die -/
def prob_less_than_five_and_even : ℚ := prob_less_than_five * prob_even

theorem prob_less_than_five_and_even_is_one_third :
  prob_less_than_five_and_even = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_five_and_even_is_one_third_l2401_240199


namespace NUMINAMATH_CALUDE_trihedral_angle_relations_l2401_240175

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  /-- Plane angles of the trihedral angle -/
  plane_angles : Fin 3 → ℝ
  /-- Dihedral angles of the trihedral angle -/
  dihedral_angles : Fin 3 → ℝ

/-- Theorem about the relationship between plane angles and dihedral angles in a trihedral angle -/
theorem trihedral_angle_relations (t : TrihedralAngle) :
  (∀ i : Fin 3, t.plane_angles i > Real.pi / 2 → ∀ j : Fin 3, t.dihedral_angles j > Real.pi / 2) ∧
  (∀ i : Fin 3, t.dihedral_angles i < Real.pi / 2 → ∀ j : Fin 3, t.plane_angles j < Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_trihedral_angle_relations_l2401_240175


namespace NUMINAMATH_CALUDE_function_properties_l2401_240104

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 + a

-- State the theorem
theorem function_properties (a : ℝ) :
  (∀ x : ℝ, f a (x + π) = f a x) ∧
  (∃ x_min : ℝ, ∀ x : ℝ, f a x_min ≤ f a x) ∧
  (∃ x_min : ℝ, f a x_min = 0) →
  (a = 1) ∧
  (∀ x : ℝ, f a x ≤ 4) ∧
  (∃ k : ℤ, ∀ x : ℝ, f a x = f a (k * π / 2 + π / 6 - x)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2401_240104


namespace NUMINAMATH_CALUDE_complex_addition_simplification_l2401_240185

theorem complex_addition_simplification :
  (-5 : ℂ) + 3*I + (2 : ℂ) - 7*I = -3 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_addition_simplification_l2401_240185


namespace NUMINAMATH_CALUDE_passing_grade_fraction_l2401_240146

theorem passing_grade_fraction (a b c d f : ℚ) : 
  a = 1/4 → b = 1/2 → c = 1/8 → d = 1/12 → f = 1/24 → a + b + c = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_passing_grade_fraction_l2401_240146


namespace NUMINAMATH_CALUDE_max_arrangement_is_eight_l2401_240191

/-- Represents a valid arrangement of balls -/
def ValidArrangement (arrangement : List Nat) : Prop :=
  (∀ n ∈ arrangement, 1 ≤ n ∧ n ≤ 9) ∧
  (5 ∈ arrangement → (arrangement.indexOf 5).pred = arrangement.indexOf 1 ∨ 
                     (arrangement.indexOf 5).succ = arrangement.indexOf 1) ∧
  (7 ∈ arrangement → (arrangement.indexOf 7).pred = arrangement.indexOf 1 ∨ 
                     (arrangement.indexOf 7).succ = arrangement.indexOf 1)

/-- The maximum number of balls that can be arranged -/
def MaxArrangement : Nat := 8

/-- Theorem stating that the maximum number of balls that can be arranged is 8 -/
theorem max_arrangement_is_eight :
  (∃ arrangement : List Nat, arrangement.length = MaxArrangement ∧ ValidArrangement arrangement) ∧
  (∀ arrangement : List Nat, arrangement.length > MaxArrangement → ¬ValidArrangement arrangement) := by
  sorry

end NUMINAMATH_CALUDE_max_arrangement_is_eight_l2401_240191


namespace NUMINAMATH_CALUDE_fifth_root_fraction_l2401_240178

theorem fifth_root_fraction : 
  (9 / 16.2) ^ (1/5 : ℝ) = (5/9 : ℝ) ^ (1/5 : ℝ) := by sorry

end NUMINAMATH_CALUDE_fifth_root_fraction_l2401_240178


namespace NUMINAMATH_CALUDE_probability_more_than_seven_is_five_twelfths_l2401_240142

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when throwing two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of favorable outcomes (totals greater than 7) -/
def favorableOutcomes : ℕ := 15

/-- The probability of getting a total more than 7 when throwing a pair of dice -/
def probabilityMoreThanSeven : ℚ := favorableOutcomes / totalOutcomes

theorem probability_more_than_seven_is_five_twelfths :
  probabilityMoreThanSeven = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_more_than_seven_is_five_twelfths_l2401_240142


namespace NUMINAMATH_CALUDE_initial_daily_steps_is_1000_l2401_240127

/-- Calculates the total steps logged over 4 weeks given the initial daily step count -/
def totalSteps (initialDailySteps : ℕ) : ℕ :=
  7 * initialDailySteps +
  7 * (initialDailySteps + 1000) +
  7 * (initialDailySteps + 2000) +
  7 * (initialDailySteps + 3000)

/-- Proves that the initial daily step count is 1000 given the problem conditions -/
theorem initial_daily_steps_is_1000 :
  ∃ (initialDailySteps : ℕ),
    totalSteps initialDailySteps = 100000 - 30000 ∧
    initialDailySteps = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_daily_steps_is_1000_l2401_240127


namespace NUMINAMATH_CALUDE_months_passed_l2401_240114

/-- Represents the number of bones Barkley receives each month -/
def bones_per_month : ℕ := 10

/-- Represents the number of bones Barkley currently has available -/
def available_bones : ℕ := 8

/-- Represents the number of bones Barkley has buried -/
def buried_bones : ℕ := 42

/-- Calculates the total number of bones Barkley has received -/
def total_bones (months : ℕ) : ℕ := bones_per_month * months

/-- Theorem stating that 5 months have passed based on the given conditions -/
theorem months_passed :
  ∃ (months : ℕ), months = 5 ∧ total_bones months = available_bones + buried_bones :=
sorry

end NUMINAMATH_CALUDE_months_passed_l2401_240114


namespace NUMINAMATH_CALUDE_equation_solution_l2401_240151

theorem equation_solution : 
  ∃ x : ℚ, (x + 5) / (x - 3) = (x - 2) / (x + 2) ∧ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2401_240151


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l2401_240105

-- Define the quadratic function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x : ℝ, f (x + 1) - f x = 2 * x - 2)
variable (h2 : f 1 = -2)
variable (h3 : ∃! (n : ℤ), f (↑n) > 0 ∧ f (↑n + t) < 0)
variable (h4 : t < 0)

-- State the theorem
theorem quadratic_function_proof :
  (∀ x : ℝ, f x = x^2 - 3*x) ∧ -2 ≤ t ∧ t < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l2401_240105


namespace NUMINAMATH_CALUDE_inequality_range_l2401_240141

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ m ∈ Set.Ioo (-10) 2 ∪ {2} :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l2401_240141


namespace NUMINAMATH_CALUDE_xyz_value_l2401_240172

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 17)
  (h3 : x^3 + y^3 + z^3 = 27) :
  x * y * z = 32 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2401_240172


namespace NUMINAMATH_CALUDE_shopkeeper_red_cards_l2401_240190

/-- Represents the number of decks for each type of playing cards --/
structure DeckCounts where
  standard : Nat
  special : Nat
  custom : Nat

/-- Represents the number of red cards in each type of deck --/
structure RedCardCounts where
  standard : Nat
  special : Nat
  custom : Nat

/-- Calculates the total number of red cards given the deck counts and red card counts --/
def totalRedCards (decks : DeckCounts) (redCards : RedCardCounts) : Nat :=
  decks.standard * redCards.standard +
  decks.special * redCards.special +
  decks.custom * redCards.custom

/-- Theorem stating that the shopkeeper has 178 red cards in total --/
theorem shopkeeper_red_cards :
  let decks : DeckCounts := { standard := 3, special := 2, custom := 2 }
  let redCards : RedCardCounts := { standard := 26, special := 30, custom := 20 }
  totalRedCards decks redCards = 178 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_red_cards_l2401_240190


namespace NUMINAMATH_CALUDE_lauryn_company_men_count_l2401_240193

theorem lauryn_company_men_count :
  ∀ (men women : ℕ),
    men + women = 180 →
    women = men + 20 →
    men = 80 := by
sorry

end NUMINAMATH_CALUDE_lauryn_company_men_count_l2401_240193


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_equals_zero_l2401_240129

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- Definition of line l₁ -/
def line_l₁ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => a * x + 2 * y - 1 = 0

/-- Definition of line l₂ -/
def line_l₂ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => 3 * x - a * y + 1 = 0

/-- The main theorem -/
theorem perpendicular_lines_a_equals_zero (a : ℝ) :
  perpendicular (a / 2) (-3 / a) → a = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_equals_zero_l2401_240129


namespace NUMINAMATH_CALUDE_converse_not_always_true_l2401_240134

theorem converse_not_always_true : ∃ (a b : ℝ), a < b ∧ ¬(∀ (m : ℝ), a * m^2 < b * m^2) :=
sorry

end NUMINAMATH_CALUDE_converse_not_always_true_l2401_240134


namespace NUMINAMATH_CALUDE_arccos_negative_half_l2401_240144

theorem arccos_negative_half : Real.arccos (-1/2) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_half_l2401_240144


namespace NUMINAMATH_CALUDE_perimeter_of_triangle_MNO_l2401_240121

/-- A right prism with equilateral triangular bases -/
structure RightPrism :=
  (height : ℝ)
  (base_side : ℝ)

/-- Points on the edges of the prism -/
structure PrismPoints (prism : RightPrism) :=
  (M : ℝ × ℝ × ℝ)
  (N : ℝ × ℝ × ℝ)
  (O : ℝ × ℝ × ℝ)

/-- The perimeter of triangle MNO in the prism -/
def triangle_perimeter (prism : RightPrism) (points : PrismPoints prism) : ℝ :=
  sorry

/-- Theorem stating the perimeter of triangle MNO -/
theorem perimeter_of_triangle_MNO (prism : RightPrism) (points : PrismPoints prism) 
  (h1 : prism.height = 20)
  (h2 : prism.base_side = 10)
  (h3 : points.M = (5, 0, 0))
  (h4 : points.N = (5, 5*Real.sqrt 3, 0))
  (h5 : points.O = (5, 0, 10)) :
  triangle_perimeter prism points = 5 + 10 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_perimeter_of_triangle_MNO_l2401_240121


namespace NUMINAMATH_CALUDE_min_distinct_values_l2401_240133

/-- Represents a list of integers with a unique mode -/
structure IntegerList where
  elements : List Nat
  mode_count : Nat
  distinct_count : Nat
  mode_is_unique : Bool

/-- Properties of the integer list -/
def valid_integer_list (L : IntegerList) : Prop :=
  L.elements.length = 2018 ∧
  L.mode_count = 10 ∧
  L.mode_is_unique = true

/-- Theorem stating the minimum number of distinct values -/
theorem min_distinct_values (L : IntegerList) :
  valid_integer_list L → L.distinct_count ≥ 225 := by
  sorry

#check min_distinct_values

end NUMINAMATH_CALUDE_min_distinct_values_l2401_240133


namespace NUMINAMATH_CALUDE_intersection_points_l2401_240108

-- Define the quadratic and linear functions
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def linear (s t x : ℝ) : ℝ := s * x + t

-- Define the discriminant
def discriminant (a b c s t : ℝ) : ℝ := (b - s)^2 - 4 * a * (c - t)

-- Theorem statement
theorem intersection_points (a b c s t : ℝ) (ha : a ≠ 0) (hs : s ≠ 0) :
  let Δ := discriminant a b c s t
  -- Two intersection points when Δ > 0
  (Δ > 0 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ quadratic a b c x₁ = linear s t x₁ ∧ quadratic a b c x₂ = linear s t x₂) ∧
  -- One intersection point when Δ = 0
  (Δ = 0 → ∃! x, quadratic a b c x = linear s t x) ∧
  -- No intersection points when Δ < 0
  (Δ < 0 → ∀ x, quadratic a b c x ≠ linear s t x) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_l2401_240108


namespace NUMINAMATH_CALUDE_ram_money_calculation_l2401_240119

theorem ram_money_calculation (ram gopal krishan : ℕ) 
  (h1 : ram * 17 = gopal * 7)
  (h2 : gopal * 17 = krishan * 7)
  (h3 : krishan = 4335) : 
  ram = 735 := by
sorry

end NUMINAMATH_CALUDE_ram_money_calculation_l2401_240119


namespace NUMINAMATH_CALUDE_calculation_proof_equation_solution_l2401_240150

-- Part 1
theorem calculation_proof :
  (Real.sqrt (25 / 9) + (Real.log 5 / Real.log 10) ^ 0 + (27 / 64) ^ (-(1/3 : ℝ))) = 4 := by
  sorry

-- Part 2
theorem equation_solution :
  ∀ x : ℝ, (Real.log (6^x - 9) / Real.log 3) = 3 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_equation_solution_l2401_240150


namespace NUMINAMATH_CALUDE_lisa_photos_l2401_240123

theorem lisa_photos (animal_photos : ℕ) 
  (h1 : animal_photos + 3 * animal_photos + (3 * animal_photos - 10) = 45) : 
  animal_photos = 7 := by
sorry

end NUMINAMATH_CALUDE_lisa_photos_l2401_240123


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2401_240163

-- Define propositions p and q
def p (x : ℝ) : Prop := 5 * x - 6 ≥ x^2
def q (x : ℝ) : Prop := |x + 1| > 2

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2401_240163


namespace NUMINAMATH_CALUDE_partition_six_into_three_l2401_240132

/-- The number of ways to partition a set of n elements into k disjoint subsets -/
def partitionWays (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to partition a set of 6 elements into 3 disjoint subsets is 15 -/
theorem partition_six_into_three : partitionWays 6 3 = 15 := by sorry

end NUMINAMATH_CALUDE_partition_six_into_three_l2401_240132


namespace NUMINAMATH_CALUDE_cubic_inequality_l2401_240107

theorem cubic_inequality (x : ℝ) : (x^3 - 125) / (x + 3) < 0 ↔ -3 < x ∧ x < 5 :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2401_240107


namespace NUMINAMATH_CALUDE_birds_ate_one_third_of_tomatoes_l2401_240138

theorem birds_ate_one_third_of_tomatoes
  (initial_tomatoes : ℕ)
  (remaining_tomatoes : ℕ)
  (h1 : initial_tomatoes = 21)
  (h2 : remaining_tomatoes = 14) :
  (initial_tomatoes - remaining_tomatoes : ℚ) / initial_tomatoes = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_birds_ate_one_third_of_tomatoes_l2401_240138


namespace NUMINAMATH_CALUDE_distribute_seven_balls_to_three_people_l2401_240162

/-- The number of ways to distribute n identical balls to k people, 
    with each person getting at least 1 ball -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 15 ways to distribute 7 identical balls to 3 people, 
    with each person getting at least 1 ball -/
theorem distribute_seven_balls_to_three_people : 
  distribute_balls 7 3 = 15 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_to_three_people_l2401_240162


namespace NUMINAMATH_CALUDE_food_duration_l2401_240177

theorem food_duration (initial_cows : ℕ) (days_passed : ℕ) (cows_left : ℕ) : 
  initial_cows = 1000 →
  days_passed = 10 →
  cows_left = 800 →
  (initial_cows * x - initial_cows * days_passed = cows_left * x) →
  x = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_food_duration_l2401_240177


namespace NUMINAMATH_CALUDE_max_value_of_g_l2401_240143

/-- Given a function f(x) = a*cos(x) + b where a and b are constants,
    if the maximum value of f(x) is 1 and the minimum value of f(x) is -7,
    then the maximum value of g(x) = a*cos(x) + b*sin(x) is 5. -/
theorem max_value_of_g (a b : ℝ) :
  (∃ x : ℝ, a * Real.cos x + b = 1) →
  (∃ x : ℝ, a * Real.cos x + b = -7) →
  (∃ x : ℝ, a * Real.cos x + b * Real.sin x = 5) ∧
  (∀ x : ℝ, a * Real.cos x + b * Real.sin x ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_g_l2401_240143


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l2401_240126

theorem polygon_angle_sum (n : ℕ) (A : ℝ) (h1 : n ≥ 3) (h2 : A > 0) :
  (n - 2) * 180 = A + 2460 →
  A = 60 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l2401_240126


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2401_240196

theorem polynomial_factorization (x : ℝ) :
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2401_240196


namespace NUMINAMATH_CALUDE_min_exercise_books_l2401_240103

def total_books : ℕ := 20
def max_cost : ℕ := 60
def exercise_book_cost : ℕ := 2
def notebook_cost : ℕ := 5

theorem min_exercise_books : 
  ∃ (x : ℕ), 
    (x ≤ total_books) ∧ 
    (exercise_book_cost * x + notebook_cost * (total_books - x) ≤ max_cost) ∧
    (∀ (y : ℕ), y < x → 
      exercise_book_cost * y + notebook_cost * (total_books - y) > max_cost) ∧
    x = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_exercise_books_l2401_240103


namespace NUMINAMATH_CALUDE_equal_values_at_fixed_distance_l2401_240154

theorem equal_values_at_fixed_distance (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc 0 1, ContinuousAt f x) →
  f 0 = 0 →
  f 1 = 0 →
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ |x₁ - x₂| = 0.1 ∧ f x₁ = f x₂ := by
  sorry


end NUMINAMATH_CALUDE_equal_values_at_fixed_distance_l2401_240154


namespace NUMINAMATH_CALUDE_negation_false_l2401_240100

theorem negation_false : ¬∃ a : ℝ, a ≤ 2 ∧ a^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_negation_false_l2401_240100


namespace NUMINAMATH_CALUDE_odd_function_property_l2401_240198

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the theorem
theorem odd_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h1 : OddFunction f) 
  (h2 : a = 2) 
  (h3 : f (-2) = 11) : 
  f a = -11 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l2401_240198


namespace NUMINAMATH_CALUDE_two_lines_two_intersections_l2401_240179

/-- The number of intersection points for n lines on a plane -/
def intersection_points (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: If n lines on a plane intersect at exactly 2 points, then n = 2 -/
theorem two_lines_two_intersections (n : ℕ) (h : intersection_points n = 2) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_lines_two_intersections_l2401_240179


namespace NUMINAMATH_CALUDE_second_division_percentage_l2401_240183

/-- Proves that the percentage of students who got second division is 54% -/
theorem second_division_percentage
  (total_students : ℕ)
  (first_division_percentage : ℚ)
  (just_passed : ℕ)
  (h_total : total_students = 300)
  (h_first : first_division_percentage = 26 / 100)
  (h_passed : just_passed = 60)
  (h_all_passed : total_students = 
    (first_division_percentage * total_students).floor + 
    (total_students - (first_division_percentage * total_students).floor - just_passed) + 
    just_passed) :
  (total_students - (first_division_percentage * total_students).floor - just_passed : ℚ) / 
  total_students * 100 = 54 := by
  sorry

end NUMINAMATH_CALUDE_second_division_percentage_l2401_240183


namespace NUMINAMATH_CALUDE_exist_distant_points_on_polyhedron_l2401_240184

/-- A sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- A polyhedron with a given number of faces -/
structure Polyhedron where
  faces : ℕ

/-- A polyhedron is circumscribed around a sphere -/
def is_circumscribed (p : Polyhedron) (s : Sphere) : Prop :=
  sorry

/-- The distance between two points on the surface of a polyhedron -/
def surface_distance (p : Polyhedron) (point1 point2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- The main theorem -/
theorem exist_distant_points_on_polyhedron (s : Sphere) (p : Polyhedron) 
  (h_radius : s.radius = 10)
  (h_faces : p.faces = 19)
  (h_circumscribed : is_circumscribed p s) :
  ∃ (point1 point2 : ℝ × ℝ × ℝ), surface_distance p point1 point2 > 21 :=
sorry

end NUMINAMATH_CALUDE_exist_distant_points_on_polyhedron_l2401_240184


namespace NUMINAMATH_CALUDE_chocolate_milk_consumption_l2401_240140

theorem chocolate_milk_consumption (milk_per_glass : ℝ) (syrup_per_glass : ℝ) 
  (total_milk : ℝ) (total_syrup : ℝ) : 
  milk_per_glass = 6.5 → 
  syrup_per_glass = 1.5 → 
  total_milk = 130 → 
  total_syrup = 60 → 
  let glasses_from_milk := total_milk / milk_per_glass
  let glasses_from_syrup := total_syrup / syrup_per_glass
  let glasses_made := min glasses_from_milk glasses_from_syrup
  let total_consumption := glasses_made * (milk_per_glass + syrup_per_glass)
  total_consumption = 160 := by
  sorry

#check chocolate_milk_consumption

end NUMINAMATH_CALUDE_chocolate_milk_consumption_l2401_240140


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2401_240186

/-- The distance between the foci of an ellipse described by the equation
    √((x-4)² + (y+5)²) + √((x+6)² + (y-7)²) = 22 is equal to 2√2. -/
theorem ellipse_foci_distance : 
  let ellipse := {p : ℝ × ℝ | Real.sqrt ((p.1 - 4)^2 + (p.2 + 5)^2) + 
                               Real.sqrt ((p.1 + 6)^2 + (p.2 - 7)^2) = 22}
  let foci := ((4, -5), (-6, 7))
  ∃ (d : ℝ), d = Real.sqrt 8 ∧ 
    d = Real.sqrt ((foci.1.1 - foci.2.1)^2 + (foci.1.2 - foci.2.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l2401_240186


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l2401_240166

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

-- Define the lines l
def line_l₁ (x y : ℝ) : Prop :=
  4*x + 3*y + 3 = 0

def line_l₂ (x y : ℝ) : Prop :=
  4*x + 3*y - 7 = 0

-- Theorem statement
theorem circle_and_line_properties :
  -- Circle C passes through O(0,0), A(-2,4), and B(1,1)
  circle_C 0 0 ∧ circle_C (-2) 4 ∧ circle_C 1 1 ∧
  -- Line l has slope -4/3
  (∀ x y : ℝ, (line_l₁ x y ∨ line_l₂ x y) → (y - 2) = -4/3 * (x + 1)) ∧
  -- The chord intercepted by circle C on line l has a length of 4
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    ((line_l₁ x₁ y₁ ∧ line_l₁ x₂ y₂) ∨ (line_l₂ x₁ y₁ ∧ line_l₂ x₂ y₂)) ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 16) →
  -- The equation of circle C is correct
  (∀ x y : ℝ, circle_C x y ↔ x^2 + y^2 + 2*x - 4*y = 0) ∧
  -- The equation of line l is one of the two given equations
  (∀ x y : ℝ, (4*x + 3*y + 3 = 0 ∨ 4*x + 3*y - 7 = 0) ↔ (line_l₁ x y ∨ line_l₂ x y)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l2401_240166


namespace NUMINAMATH_CALUDE_distance_between_parallel_lines_l2401_240189

/-- The distance between two parallel lines in R² --/
theorem distance_between_parallel_lines :
  let line1 : ℝ → ℝ × ℝ := λ t ↦ (4 + 2*t, -1 - 6*t)
  let line2 : ℝ → ℝ × ℝ := λ s ↦ (3 + 2*s, -2 - 6*s)
  let v : ℝ × ℝ := (3 - 4, -2 - (-1))
  let d : ℝ × ℝ := (2, -6)
  let distance := ‖v - (((v.1 * d.1 + v.2 * d.2) / (d.1^2 + d.2^2)) • d)‖
  distance = 2 * Real.sqrt 10 / 5 := by
sorry


end NUMINAMATH_CALUDE_distance_between_parallel_lines_l2401_240189


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l2401_240161

theorem consecutive_even_integers_sum (x : ℕ) (h : x > 0) :
  (x - 2) * x * (x + 2) = 20 * ((x - 2) + x + (x + 2)) →
  (x - 2) + x + (x + 2) = 24 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l2401_240161


namespace NUMINAMATH_CALUDE_sqrt_3_minus_1_over_2_less_than_half_l2401_240116

theorem sqrt_3_minus_1_over_2_less_than_half : (Real.sqrt 3 - 1) / 2 < 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_minus_1_over_2_less_than_half_l2401_240116


namespace NUMINAMATH_CALUDE_classification_theorem_l2401_240106

def expressions : List String := [
  "4xy", "m^2n/2", "y^2 + y + 2/y", "2x^3 - 3", "0", "-3/(ab) + a",
  "m", "(m-n)/(m+n)", "(x-1)/2", "3/x"
]

def is_monomial (expr : String) : Bool := sorry

def is_polynomial (expr : String) : Bool := sorry

theorem classification_theorem :
  let monomials := expressions.filter is_monomial
  let polynomials := expressions.filter (λ e => is_polynomial e ∧ ¬is_monomial e)
  let all_polynomials := expressions.filter is_polynomial
  (monomials = ["4xy", "m^2n/2", "0", "m"]) ∧
  (polynomials = ["2x^3 - 3", "(x-1)/2"]) ∧
  (all_polynomials = ["4xy", "m^2n/2", "2x^3 - 3", "0", "m", "(x-1)/2"]) := by
  sorry

end NUMINAMATH_CALUDE_classification_theorem_l2401_240106
