import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l2960_296003

theorem problem_statement (x y : ℝ) (h : -y + 3*x = 3) : 
  2*(y - 3*x) - (3*x - y)^2 + 1 = -14 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2960_296003


namespace NUMINAMATH_CALUDE_roy_has_114_pens_l2960_296002

/-- The number of pens Roy has -/
structure PenCounts where
  blue : ℕ
  black : ℕ
  red : ℕ
  green : ℕ
  purple : ℕ

/-- Roy's pen collection satisfies the given conditions -/
def satisfiesConditions (p : PenCounts) : Prop :=
  p.blue = 8 ∧
  p.black = 4 * p.blue ∧
  p.red = p.blue + p.black - 5 ∧
  p.green = p.red / 2 ∧
  p.purple = p.blue + p.green - 3

/-- The total number of pens Roy has -/
def totalPens (p : PenCounts) : ℕ :=
  p.blue + p.black + p.red + p.green + p.purple

/-- Theorem: Roy has 114 pens in total -/
theorem roy_has_114_pens :
  ∃ p : PenCounts, satisfiesConditions p ∧ totalPens p = 114 := by
  sorry

end NUMINAMATH_CALUDE_roy_has_114_pens_l2960_296002


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l2960_296097

theorem sin_cos_difference_equals_half : 
  Real.sin (65 * π / 180) * Real.cos (35 * π / 180) - 
  Real.sin (25 * π / 180) * Real.sin (35 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l2960_296097


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l2960_296046

theorem factorial_equation_solution :
  ∃! (a b : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
  (5 * 4 * 3 * 2 * 1)^8 + (5 * 4 * 3 * 2 * 1)^7 = 4000000000000000 + a * 100000000000000 + 356000000000000 + 400000000000 + 80000000000 + b * 10000000000 + 80000000000 ∧
  a = 3 ∧ b = 6 := by
sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l2960_296046


namespace NUMINAMATH_CALUDE_exists_composite_in_sequence_l2960_296039

-- Define the sequence type
def RecurrenceSequence := ℕ → ℕ

-- Define the recurrence relation
def SatisfiesRecurrence (a : RecurrenceSequence) : Prop :=
  ∀ n : ℕ, (a (n + 1) = 2 * a n + 1) ∨ (a (n + 1) = 2 * a n - 1)

-- Define a non-constant sequence
def NonConstant (a : RecurrenceSequence) : Prop :=
  ∃ m n : ℕ, a m ≠ a n

-- Define a positive sequence
def Positive (a : RecurrenceSequence) : Prop :=
  ∀ n : ℕ, a n > 0

-- Define a composite number
def Composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

-- The main theorem
theorem exists_composite_in_sequence (a : RecurrenceSequence)
  (h1 : SatisfiesRecurrence a)
  (h2 : NonConstant a)
  (h3 : Positive a) :
  ∃ n : ℕ, Composite (a n) :=
  sorry

end NUMINAMATH_CALUDE_exists_composite_in_sequence_l2960_296039


namespace NUMINAMATH_CALUDE_complex_number_problem_l2960_296079

theorem complex_number_problem (a : ℝ) : 
  (∃ (z₁ : ℂ), z₁ = a + (2 / (1 - Complex.I)) ∧ z₁.re < 0 ∧ z₁.im > 0) ∧ 
  Complex.abs (a - Complex.I) = 2 → 
  a = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2960_296079


namespace NUMINAMATH_CALUDE_star_properties_l2960_296071

/-- Custom multiplication operation -/
def star (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating that exactly two of the given properties hold for the star operation -/
theorem star_properties :
  (∃! n : ℕ, n = 2 ∧ 
    (((∀ a b : ℝ, star a b = 0 → a = 0 ∧ b = 0) → n ≥ 1) ∧
     ((∀ a b : ℝ, star a b = star b a) → n ≥ 1) ∧
     ((∀ a b c : ℝ, star a (b + c) = star a b + star a c) → n ≥ 1) ∧
     ((∀ a b : ℝ, star a b = star (-a) (-b)) → n ≥ 1)) ∧
    n ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_star_properties_l2960_296071


namespace NUMINAMATH_CALUDE_smallest_student_group_l2960_296037

theorem smallest_student_group (n : ℕ) : n = 46 ↔ 
  (n > 0) ∧ 
  (n % 3 = 1) ∧ 
  (n % 6 = 4) ∧ 
  (n % 8 = 5) ∧ 
  (∀ m : ℕ, m > 0 → m % 3 = 1 → m % 6 = 4 → m % 8 = 5 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_student_group_l2960_296037


namespace NUMINAMATH_CALUDE_two_eyes_for_dog_l2960_296082

/-- Given a family that catches and distributes fish, calculate the number of fish eyes left for the dog. -/
def fish_eyes_for_dog (family_size : ℕ) (fish_per_person : ℕ) (eyes_per_fish : ℕ) (eyes_eaten : ℕ) : ℕ :=
  let total_fish := family_size * fish_per_person
  let total_eyes := total_fish * eyes_per_fish
  total_eyes - eyes_eaten

/-- Theorem stating that under the given conditions, 2 fish eyes remain for the dog. -/
theorem two_eyes_for_dog :
  fish_eyes_for_dog 3 4 2 22 = 2 :=
by sorry

end NUMINAMATH_CALUDE_two_eyes_for_dog_l2960_296082


namespace NUMINAMATH_CALUDE_trajectory_properties_line_intersection_condition_unique_k_for_dot_product_l2960_296006

noncomputable def trajectory (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y - 1)^2) = Real.sqrt 2 * Real.sqrt ((x - 1)^2 + (y - 2)^2)

def line_intersects (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x - 1

theorem trajectory_properties :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, trajectory x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    center_x = 2 ∧ center_y = 3 ∧ radius = 2 :=
sorry

theorem line_intersection_condition (k : ℝ) :
  (∃ x y, trajectory x y ∧ line_intersects k x y) ↔ k > 3/4 :=
sorry

theorem unique_k_for_dot_product :
  ∃! k, k > 3/4 ∧
    ∀ x₁ y₁ x₂ y₂,
      trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
      line_intersects k x₁ y₁ ∧ line_intersects k x₂ y₂ →
      x₁ * x₂ + y₁ * y₂ = 11 :=
sorry

end NUMINAMATH_CALUDE_trajectory_properties_line_intersection_condition_unique_k_for_dot_product_l2960_296006


namespace NUMINAMATH_CALUDE_relationship_abc_l2960_296058

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 4 then 4 / x + 1 else Real.log x / Real.log 2

theorem relationship_abc (a b c : ℝ) :
  (0 < a ∧ a < 4) →
  (b ≥ 4) →
  (f a = c) →
  (f b = c) →
  (deriv f b < 0) →
  b > a ∧ a > c :=
sorry

end NUMINAMATH_CALUDE_relationship_abc_l2960_296058


namespace NUMINAMATH_CALUDE_projects_equal_volume_projects_equal_days_l2960_296055

/-- Represents the dimensions of an excavation project -/
structure ProjectDimensions where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of earth to be dug given project dimensions -/
def calculateVolume (dimensions : ProjectDimensions) : ℝ :=
  dimensions.depth * dimensions.length * dimensions.breadth

/-- The dimensions of Project 1 -/
def project1 : ProjectDimensions := {
  depth := 100,
  length := 25,
  breadth := 30
}

/-- The dimensions of Project 2 -/
def project2 : ProjectDimensions := {
  depth := 75,
  length := 20,
  breadth := 50
}

/-- Theorem stating that the volumes of both projects are equal -/
theorem projects_equal_volume : calculateVolume project1 = calculateVolume project2 := by
  sorry

/-- Corollary stating that the number of days required for both projects is the same -/
theorem projects_equal_days (days1 days2 : ℕ) 
    (h : calculateVolume project1 = calculateVolume project2) : days1 = days2 := by
  sorry

end NUMINAMATH_CALUDE_projects_equal_volume_projects_equal_days_l2960_296055


namespace NUMINAMATH_CALUDE_complex_magnitude_of_special_z_l2960_296089

theorem complex_magnitude_of_special_z : 
  let i : ℂ := Complex.I
  let z : ℂ := -i^2022 + i
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_of_special_z_l2960_296089


namespace NUMINAMATH_CALUDE_expression_evaluation_l2960_296070

theorem expression_evaluation : (20 ^ 40) / (40 ^ 20) = 10 ^ 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2960_296070


namespace NUMINAMATH_CALUDE_shortest_path_equals_two_R_l2960_296031

/-- A truncated cone with a specific angle between generatrix and larger base -/
structure TruncatedCone where
  R : ℝ  -- Radius of the larger base
  r : ℝ  -- Radius of the smaller base
  h : ℝ  -- Height of the truncated cone
  angle : ℝ  -- Angle between generatrix and larger base in radians

/-- The shortest path on the surface of a truncated cone -/
def shortestPath (cone : TruncatedCone) : ℝ := sorry

/-- Theorem stating that the shortest path is twice the radius of the larger base -/
theorem shortest_path_equals_two_R (cone : TruncatedCone) 
  (h₁ : cone.angle = π / 3)  -- 60 degrees in radians
  : shortestPath cone = 2 * cone.R := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_equals_two_R_l2960_296031


namespace NUMINAMATH_CALUDE_specific_pentagon_area_l2960_296000

/-- Pentagon with specified side lengths and right angles -/
structure Pentagon where
  PQ : ℝ
  QR : ℝ
  RS : ℝ
  ST : ℝ
  TP : ℝ
  angle_TPQ : ℝ
  angle_PQR : ℝ

/-- The area of a pentagon with the given properties -/
def pentagon_area (p : Pentagon) : ℝ := sorry

/-- Theorem stating the area of the specific pentagon is 100 -/
theorem specific_pentagon_area :
  let p : Pentagon := {
    PQ := 8,
    QR := 2,
    RS := 13,
    ST := 13,
    TP := 8,
    angle_TPQ := 90,
    angle_PQR := 90
  }
  pentagon_area p = 100 := by sorry

end NUMINAMATH_CALUDE_specific_pentagon_area_l2960_296000


namespace NUMINAMATH_CALUDE_base_conversion_sum_l2960_296013

/-- Converts a number from given base to base 10 -/
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

/-- The main theorem -/
theorem base_conversion_sum :
  let a := to_base_10 [2, 1, 3] 8
  let b := to_base_10 [1, 2] 3
  let c := to_base_10 [2, 3, 4] 5
  let d := to_base_10 [3, 2] 4
  (a / b : Rat) + (c / d : Rat) = 31 + 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l2960_296013


namespace NUMINAMATH_CALUDE_cleaning_time_theorem_l2960_296051

/-- Represents the grove of trees -/
structure Grove :=
  (rows : ℕ)
  (columns : ℕ)

/-- Calculates the time to clean each tree without help -/
def time_per_tree_without_help (g : Grove) (total_time_with_help : ℕ) : ℚ :=
  let total_trees := g.rows * g.columns
  let time_per_tree_with_help := total_time_with_help / total_trees
  2 * time_per_tree_with_help

theorem cleaning_time_theorem (g : Grove) (h : g.rows = 4 ∧ g.columns = 5) :
  time_per_tree_without_help g 60 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_time_theorem_l2960_296051


namespace NUMINAMATH_CALUDE_triangle_properties_l2960_296026

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real

-- State the theorem
theorem triangle_properties (t : Triangle)
  (h1 : t.A + t.B + t.C = π)
  (h2 : t.S > 0)
  (h3 : Real.tan (t.A / 2) * Real.tan (t.B / 2) + Real.sqrt 3 * (Real.tan (t.A / 2) + Real.tan (t.B / 2)) = 1) :
  t.C = 2 * π / 3 ∧ t.c^2 ≥ 4 * Real.sqrt 3 * t.S := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2960_296026


namespace NUMINAMATH_CALUDE_wong_grandchildren_probability_l2960_296004

/-- The number of grandchildren Mr. Wong has -/
def num_grandchildren : ℕ := 12

/-- The probability of a grandchild being male (or female) -/
def gender_probability : ℚ := 1/2

/-- The probability of having an unequal number of grandsons and granddaughters -/
def unequal_gender_prob : ℚ := 793/1024

theorem wong_grandchildren_probability :
  let total_outcomes := 2^num_grandchildren
  let equal_gender_outcomes := (num_grandchildren.choose (num_grandchildren / 2))
  (total_outcomes - equal_gender_outcomes : ℚ) / total_outcomes = unequal_gender_prob :=
sorry

end NUMINAMATH_CALUDE_wong_grandchildren_probability_l2960_296004


namespace NUMINAMATH_CALUDE_g_of_3_equals_5_l2960_296095

-- Define the function g
def g (x : ℝ) : ℝ := 2 * (x - 2) + 3

-- State the theorem
theorem g_of_3_equals_5 : g 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_5_l2960_296095


namespace NUMINAMATH_CALUDE_library_visitors_average_l2960_296060

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (days_in_month : ℕ) (h1 : sunday_visitors = 510) (h2 : other_day_visitors = 240) 
  (h3 : days_in_month = 30) : 
  (5 * sunday_visitors + 25 * other_day_visitors) / days_in_month = 285 :=
by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l2960_296060


namespace NUMINAMATH_CALUDE_min_score_for_higher_average_l2960_296030

/-- Represents the scores of a student in four tests -/
structure Scores :=
  (test1 : ℕ) (test2 : ℕ) (test3 : ℕ) (test4 : ℕ)

/-- A-Long's scores -/
def aLong : Scores :=
  { test1 := 81, test2 := 81, test3 := 81, test4 := 81 }

/-- A-Hai's scores -/
def aHai : Scores :=
  { test1 := aLong.test1 + 1,
    test2 := aLong.test2 + 2,
    test3 := aLong.test3 + 3,
    test4 := 99 }

/-- The average score of a student -/
def average (s : Scores) : ℚ :=
  (s.test1 + s.test2 + s.test3 + s.test4) / 4

theorem min_score_for_higher_average :
  average aHai ≥ average aLong + 4 :=
by sorry

end NUMINAMATH_CALUDE_min_score_for_higher_average_l2960_296030


namespace NUMINAMATH_CALUDE_sin_sum_inverse_trig_functions_l2960_296091

theorem sin_sum_inverse_trig_functions :
  Real.sin (Real.arcsin (4/5) + Real.arctan (3/2) + Real.arccos (1/3)) = (17 - 12 * Real.sqrt 2) / (15 * Real.sqrt 13) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_inverse_trig_functions_l2960_296091


namespace NUMINAMATH_CALUDE_marta_textbook_expenses_l2960_296015

/-- The total amount Marta spent on textbooks -/
def total_spent (sale_price : ℕ) (sale_quantity : ℕ) (online_total : ℕ) (bookstore_multiplier : ℕ) : ℕ :=
  sale_price * sale_quantity + online_total + bookstore_multiplier * online_total

/-- Theorem stating the total amount Marta spent on textbooks -/
theorem marta_textbook_expenses : total_spent 10 5 40 3 = 210 := by
  sorry

end NUMINAMATH_CALUDE_marta_textbook_expenses_l2960_296015


namespace NUMINAMATH_CALUDE_cubic_equation_implies_specific_value_l2960_296043

theorem cubic_equation_implies_specific_value :
  ∀ x : ℝ, x^3 - 3 * Real.sqrt 2 * x^2 + 6 * x - 2 * Real.sqrt 2 - 8 = 0 →
  x^5 - 41 * x^2 + 2012 = 1998 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_implies_specific_value_l2960_296043


namespace NUMINAMATH_CALUDE_integer_solutions_count_l2960_296092

/-- The number of distinct integer values of a for which x^2 + ax + 9a = 0 has integer solutions for x -/
theorem integer_solutions_count : 
  (∃ (S : Finset ℤ), (∀ a : ℤ, (∃ x : ℤ, x^2 + a*x + 9*a = 0) ↔ a ∈ S) ∧ Finset.card S = 5) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_count_l2960_296092


namespace NUMINAMATH_CALUDE_tan_negative_405_degrees_l2960_296059

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_405_degrees_l2960_296059


namespace NUMINAMATH_CALUDE_hawthorn_box_maximum_l2960_296047

theorem hawthorn_box_maximum (N : ℕ) : 
  N > 100 ∧
  N % 3 = 1 ∧
  N % 4 = 2 ∧
  N % 5 = 3 ∧
  N % 6 = 4 →
  N ≤ 178 ∧ ∃ (M : ℕ), M = 178 ∧ 
    M > 100 ∧
    M % 3 = 1 ∧
    M % 4 = 2 ∧
    M % 5 = 3 ∧
    M % 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hawthorn_box_maximum_l2960_296047


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2960_296044

/-- For a quadratic equation ax^2 + 2x + 1 = 0 to have real roots, 
    a must satisfy: a ≤ 1 and a ≠ 0 -/
theorem quadratic_real_roots_condition (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) ↔ (a ≤ 1 ∧ a ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2960_296044


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_5_m_range_when_solution_set_is_real_l2960_296040

def f (x m : ℝ) : ℝ := |x + 1| + |x - 2| - m

theorem solution_set_when_m_is_5 :
  {x : ℝ | f x 5 > 0} = {x : ℝ | x < -2 ∨ x > 3} :=
sorry

theorem m_range_when_solution_set_is_real :
  (∀ x : ℝ, f x m ≥ 2) → m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_5_m_range_when_solution_set_is_real_l2960_296040


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l2960_296019

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  4 * x^4 + 16 * y^4 + 36 * z^4 + 9 / (x * y * z) ≥ 24 :=
by
  sorry

theorem min_value_attainable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  4 * x^4 + 16 * y^4 + 36 * z^4 + 9 / (x * y * z) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l2960_296019


namespace NUMINAMATH_CALUDE_square_intersection_perimeter_l2960_296066

/-- Given a square with side length 2a and a line y = -x/3 intersecting it,
    the perimeter of one resulting quadrilateral divided by a is (8 + 2√10) / 3 -/
theorem square_intersection_perimeter (a : ℝ) (a_pos : a > 0) :
  let square_vertices := [(-a, -a), (a, -a), (-a, a), (a, a)]
  let intersecting_line (x : ℝ) := -x/3
  let intersection_points := [(-a, a/3), (a, -a/3)]
  let quadrilateral_vertices := [(-a, a), (-a, a/3), (a, -a/3), (a, -a)]
  let perimeter := 
    2 * (a - a/3) +  -- vertical sides
    2 * a +          -- horizontal side
    Real.sqrt ((2*a)^2 + (2*a/3)^2)  -- diagonal
  perimeter / a = (8 + 2 * Real.sqrt 10) / 3 := by
  sorry


end NUMINAMATH_CALUDE_square_intersection_perimeter_l2960_296066


namespace NUMINAMATH_CALUDE_vacuum_cleaner_theorem_l2960_296005

def vacuum_cleaner_problem (initial_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) 
  (dog_walking_earnings : List ℕ) (discount_percent : ℕ) : ℕ × ℕ :=
  let discounted_cost := initial_cost - (initial_cost * discount_percent / 100)
  let total_savings := initial_savings + weekly_allowance * 3 + dog_walking_earnings.sum
  let amount_needed := discounted_cost - total_savings
  let weekly_savings := weekly_allowance + dog_walking_earnings.getLast!
  let weeks_needed := (amount_needed + weekly_savings - 1) / weekly_savings
  (amount_needed, weeks_needed)

theorem vacuum_cleaner_theorem : 
  vacuum_cleaner_problem 420 65 25 [40, 50, 30] 15 = (97, 2) := by
  sorry

end NUMINAMATH_CALUDE_vacuum_cleaner_theorem_l2960_296005


namespace NUMINAMATH_CALUDE_negative_x_exponent_product_l2960_296067

theorem negative_x_exponent_product (x : ℝ) : (-x)^3 * (-x)^4 = -x^7 := by sorry

end NUMINAMATH_CALUDE_negative_x_exponent_product_l2960_296067


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l2960_296020

theorem product_zero_implies_factor_zero (a b c : ℝ) : a * b * c = 0 → (a = 0 ∨ b = 0 ∨ c = 0) := by
  sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l2960_296020


namespace NUMINAMATH_CALUDE_area_is_24_l2960_296054

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := |3 * x| + |4 * y| = 12

/-- The graph is symmetric with respect to both x-axis and y-axis -/
axiom symmetry : ∀ x y : ℝ, equation x y ↔ equation (-x) y ∧ equation x (-y)

/-- The area enclosed by the graph -/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the enclosed area is 24 square units -/
theorem area_is_24 : enclosed_area = 24 :=
sorry

end NUMINAMATH_CALUDE_area_is_24_l2960_296054


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2960_296008

/-- Given a principal P put at simple interest for 3 years, if increasing the interest rate by 2% 
    results in Rs. 360 more interest, then P = 6000. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 2) * 3) / 100 = (P * R * 3) / 100 + 360 → P = 6000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2960_296008


namespace NUMINAMATH_CALUDE_monthly_income_problem_l2960_296076

/-- Given the average monthly incomes of three people, prove the income of one person -/
theorem monthly_income_problem (A B C : ℝ) 
  (h1 : (A + B) / 2 = 4050)
  (h2 : (B + C) / 2 = 5250)
  (h3 : (A + C) / 2 = 4200) :
  A = 3000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_problem_l2960_296076


namespace NUMINAMATH_CALUDE_sum_difference_equals_product_l2960_296009

-- Define the sequence
def seq : ℕ → ℕ
  | 0 => 0
  | n + 1 => n / 2 + 1

-- Define f(n) as the sum of the first n terms of the sequence
def f (n : ℕ) : ℕ := (List.range n).map seq |>.sum

-- Theorem statement
theorem sum_difference_equals_product {s t : ℕ} (hs : s > 0) (ht : t > 0) (hst : s > t) :
  f (s + t) - f (s - t) = s * t := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equals_product_l2960_296009


namespace NUMINAMATH_CALUDE_beths_bowling_score_l2960_296035

/-- Given the bowling scores of Gretchen and Mitzi, and the average score of all three bowlers,
    calculate Beth's bowling score. -/
theorem beths_bowling_score
  (gretchens_score : ℕ)
  (mitzis_score : ℕ)
  (average_score : ℕ)
  (h1 : gretchens_score = 120)
  (h2 : mitzis_score = 113)
  (h3 : average_score = 106)
  (h4 : (gretchens_score + mitzis_score + beths_score) / 3 = average_score) :
  beths_score = 85 :=
by
  sorry

#check beths_bowling_score

end NUMINAMATH_CALUDE_beths_bowling_score_l2960_296035


namespace NUMINAMATH_CALUDE_divisible_by_4_or_6_count_l2960_296041

def count_divisible (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

theorem divisible_by_4_or_6_count :
  (count_divisible 51 4) + (count_divisible 51 6) - (count_divisible 51 12) = 16 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_4_or_6_count_l2960_296041


namespace NUMINAMATH_CALUDE_tangent_triangle_angles_correct_l2960_296065

structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_angles : α + β + γ = Real.pi
  not_right : α ≠ Real.pi/2 ∧ β ≠ Real.pi/2 ∧ γ ≠ Real.pi/2

def tangent_triangle_angles (t : Triangle) : Set Real :=
  if t.α < Real.pi/2 ∧ t.β < Real.pi/2 ∧ t.γ < Real.pi/2 then
    {Real.pi - 2*t.α, Real.pi - 2*t.β, Real.pi - 2*t.γ}
  else
    {2*t.α - Real.pi, 2*t.γ, 2*t.β}

theorem tangent_triangle_angles_correct (t : Triangle) :
  ∃ (a b c : Real), tangent_triangle_angles t = {a, b, c} ∧ a + b + c = Real.pi :=
sorry

end NUMINAMATH_CALUDE_tangent_triangle_angles_correct_l2960_296065


namespace NUMINAMATH_CALUDE_drivers_days_off_l2960_296025

/-- Proves that drivers get 5 days off per month given the specified conditions -/
theorem drivers_days_off 
  (num_drivers : ℕ) 
  (days_in_month : ℕ) 
  (total_cars : ℕ) 
  (maintenance_percentage : ℚ) 
  (h1 : num_drivers = 54)
  (h2 : days_in_month = 30)
  (h3 : total_cars = 60)
  (h4 : maintenance_percentage = 1/4) : 
  (days_in_month : ℚ) - (total_cars * (1 - maintenance_percentage) * days_in_month) / num_drivers = 5 := by
  sorry

end NUMINAMATH_CALUDE_drivers_days_off_l2960_296025


namespace NUMINAMATH_CALUDE_product_inequality_l2960_296061

theorem product_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (2 + x) * (2 + y) * (2 + z) ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l2960_296061


namespace NUMINAMATH_CALUDE_derivative_of_one_minus_cosine_l2960_296001

theorem derivative_of_one_minus_cosine (α : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 1 - Real.cos x
  (deriv f) α = Real.sin α := by
sorry

end NUMINAMATH_CALUDE_derivative_of_one_minus_cosine_l2960_296001


namespace NUMINAMATH_CALUDE_solution_set_characterization_range_of_a_characterization_l2960_296099

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - 2)

-- Part 1: Characterize the solution set of f(x) ≥ 4
theorem solution_set_characterization :
  {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -4/3 ∨ x ≥ 0} :=
sorry

-- Part 2: Characterize the range of a
theorem range_of_a_characterization :
  {a : ℝ | ∀ x > 0, f x + a * x - 1 > 0} = {a : ℝ | a > -5/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_characterization_range_of_a_characterization_l2960_296099


namespace NUMINAMATH_CALUDE_fraction_inequality_l2960_296052

theorem fraction_inequality (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : (a : ℝ) / b > Real.sqrt 2) :
  (a : ℝ) / b - 1 / (2 * (a : ℝ) * b) > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2960_296052


namespace NUMINAMATH_CALUDE_water_ratio_horse_to_pig_l2960_296094

/-- Proves that the ratio of water needed by a horse to a pig is 2:1 given the specified conditions -/
theorem water_ratio_horse_to_pig :
  let num_pigs : ℕ := 8
  let num_horses : ℕ := 10
  let water_per_pig : ℕ := 3
  let water_for_chickens : ℕ := 30
  let total_water : ℕ := 114
  let water_for_horses : ℕ := total_water - (num_pigs * water_per_pig) - water_for_chickens
  let water_per_horse : ℚ := water_for_horses / num_horses
  water_per_horse / water_per_pig = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_water_ratio_horse_to_pig_l2960_296094


namespace NUMINAMATH_CALUDE_quebec_temperature_l2960_296075

-- Define the temperatures as integers (assuming we're working with whole numbers)
def temp_vancouver : ℤ := 22
def temp_calgary : ℤ := temp_vancouver - 19
def temp_quebec : ℤ := temp_calgary - 11

-- Theorem to prove
theorem quebec_temperature : temp_quebec = -8 := by
  sorry

end NUMINAMATH_CALUDE_quebec_temperature_l2960_296075


namespace NUMINAMATH_CALUDE_fraction_equality_l2960_296028

theorem fraction_equality (x : ℝ) : (5 + x) / (7 + x) = (3 + x) / (4 + x) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2960_296028


namespace NUMINAMATH_CALUDE_drum_oil_capacity_l2960_296078

theorem drum_oil_capacity (c : ℝ) (h1 : c > 0) : 
  let drum_x_capacity := c
  let drum_x_oil := (1 / 2 : ℝ) * drum_x_capacity
  let drum_y_capacity := 2 * drum_x_capacity
  let drum_y_oil := (1 / 3 : ℝ) * drum_y_capacity
  let final_oil := drum_y_oil + drum_x_oil
  final_oil / drum_y_capacity = 7 / 12
  := by sorry

end NUMINAMATH_CALUDE_drum_oil_capacity_l2960_296078


namespace NUMINAMATH_CALUDE_investment_growth_l2960_296096

/-- Calculates the final amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Proves that the given investment scenario results in the expected amount -/
theorem investment_growth (principal : ℝ) (rate : ℝ) (years : ℕ) 
  (h1 : principal = 2000)
  (h2 : rate = 0.05)
  (h3 : years = 18) :
  ∃ ε > 0, |compound_interest principal rate years - 4813.24| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_growth_l2960_296096


namespace NUMINAMATH_CALUDE_arccos_zero_l2960_296024

theorem arccos_zero : Real.arccos 0 = π / 2 := by sorry

end NUMINAMATH_CALUDE_arccos_zero_l2960_296024


namespace NUMINAMATH_CALUDE_angle_PSU_is_20_degrees_l2960_296072

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the angle measure in degrees
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the foot of the perpendicular
def foot_of_perpendicular (P S Q R : ℝ × ℝ) : Prop :=
  sorry

-- Define the center of the circumscribed circle
def circumcenter (T P Q R : ℝ × ℝ) : Prop :=
  sorry

-- Define a point on the diameter opposite to another point
def opposite_on_diameter (P U T : ℝ × ℝ) : Prop :=
  sorry

theorem angle_PSU_is_20_degrees 
  (P Q R S T U : ℝ × ℝ) 
  (triangle : Triangle P Q R)
  (angle_PRQ : angle_measure P R Q = 60)
  (angle_QRP : angle_measure Q R P = 80)
  (S_perpendicular : foot_of_perpendicular P S Q R)
  (T_circumcenter : circumcenter T P Q R)
  (U_opposite : opposite_on_diameter P U T) :
  angle_measure P S U = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_PSU_is_20_degrees_l2960_296072


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2960_296021

theorem functional_equation_solution (f : ℤ → ℤ) :
  (∀ m n : ℤ, f (f m + n) + f m = f n + f (3 * m) + 2014) →
  ∃ c : ℤ, ∀ m : ℤ, f m = 2 * m + c :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2960_296021


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2960_296088

theorem necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a > 2 ∧ b > 2 → a + b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2960_296088


namespace NUMINAMATH_CALUDE_defective_product_probability_l2960_296056

/-- The probability of drawing a defective product on the second draw,
    given that the first draw was a defective product, when there are
    10 total products, 4 of which are defective, and 2 products are
    drawn successively without replacement. -/
theorem defective_product_probability :
  let total_products : ℕ := 10
  let defective_products : ℕ := 4
  let qualified_products : ℕ := total_products - defective_products
  let first_draw_defective_prob : ℚ := defective_products / total_products
  let second_draw_defective_prob : ℚ :=
    (defective_products - 1) / (total_products - 1)
  let conditional_prob : ℚ :=
    (first_draw_defective_prob * second_draw_defective_prob) / first_draw_defective_prob
  conditional_prob = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_defective_product_probability_l2960_296056


namespace NUMINAMATH_CALUDE_certain_value_proof_l2960_296077

theorem certain_value_proof (x w : ℝ) (h1 : 13 = x / (1 - w)) (h2 : w^2 = 1) : x = 26 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_proof_l2960_296077


namespace NUMINAMATH_CALUDE_ivan_age_l2960_296057

/-- Calculates the total complete years given an age in various units -/
def total_complete_years (years months weeks days hours : ℕ) : ℕ :=
  let months_to_years := months / 12
  let weeks_to_days := weeks * 7
  let total_days := weeks_to_days + days
  let days_to_years := total_days / 365
  let hours_to_days := hours / 24
  years + months_to_years + days_to_years

/-- Theorem stating that given age of 48 years, 48 months, 48 weeks, 48 days, and 48 hours results in 53 complete years -/
theorem ivan_age : total_complete_years 48 48 48 48 48 = 53 := by
  sorry

#eval total_complete_years 48 48 48 48 48

end NUMINAMATH_CALUDE_ivan_age_l2960_296057


namespace NUMINAMATH_CALUDE_red_crayons_per_person_l2960_296073

def initial_rulers : ℕ := 11
def initial_crayons : ℕ := 34
def tim_added_rulers : ℕ := 14
def jane_removed_crayons : ℕ := 20
def jane_added_blue_crayons : ℕ := 8
def number_of_people : ℕ := 3

def total_red_crayons : ℕ := 2 * jane_added_blue_crayons

theorem red_crayons_per_person :
  total_red_crayons / number_of_people = 5 := by sorry

end NUMINAMATH_CALUDE_red_crayons_per_person_l2960_296073


namespace NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_problem_four_l2960_296017

-- Problem 1
theorem problem_one : (-23) - (-58) + (-17) = 18 := by sorry

-- Problem 2
theorem problem_two : (-8) / (-1 - 1/9) * 0.125 = 9/10 := by sorry

-- Problem 3
theorem problem_three : (-1/3 - 1/4 + 1/15) * (-60) = 31 := by sorry

-- Problem 4
theorem problem_four : -1^2 * |(-1/4)| + (-1/2)^3 / (-1)^2023 = -1/8 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_problem_four_l2960_296017


namespace NUMINAMATH_CALUDE_proportion_problem_l2960_296007

theorem proportion_problem : ∃ X : ℝ, (8 / 4 = X / 240) ∧ X = 480 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l2960_296007


namespace NUMINAMATH_CALUDE_special_number_exists_l2960_296023

def digit_product (n : ℕ) : ℕ := sorry

def digit_sum (n : ℕ) : ℕ := sorry

def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem special_number_exists : ∃ x : ℕ, 
  (digit_product x = 44 * x - 86868) ∧ 
  (is_cube (digit_sum x)) := by sorry

end NUMINAMATH_CALUDE_special_number_exists_l2960_296023


namespace NUMINAMATH_CALUDE_radio_price_calculation_l2960_296010

/-- Given a radio with 7% sales tax, if reducing its price by 161.46 results in a price of 2468,
    then the original price including sales tax is 2629.46. -/
theorem radio_price_calculation (original_price : ℝ) : 
  (original_price - 161.46 = 2468) → original_price = 2629.46 := by
  sorry

end NUMINAMATH_CALUDE_radio_price_calculation_l2960_296010


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2960_296080

/-- Proves that mixing 300 mL of 10% alcohol solution with 200 mL of 30% alcohol solution results in 18% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 300
  let y_volume : ℝ := 200
  let x_concentration : ℝ := 0.10
  let y_concentration : ℝ := 0.30
  let final_concentration : ℝ := 0.18
  (x_volume * x_concentration + y_volume * y_concentration) / (x_volume + y_volume) = final_concentration := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2960_296080


namespace NUMINAMATH_CALUDE_average_price_of_books_l2960_296084

/-- The average price of books bought by Rahim -/
theorem average_price_of_books (books_shop1 : ℕ) (price_shop1 : ℕ) 
  (books_shop2 : ℕ) (price_shop2 : ℕ) :
  books_shop1 = 40 →
  price_shop1 = 600 →
  books_shop2 = 20 →
  price_shop2 = 240 →
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2) = 14 := by
  sorry

#check average_price_of_books

end NUMINAMATH_CALUDE_average_price_of_books_l2960_296084


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l2960_296018

/-- Given two employees X and Y with a total pay of 330 and Y's pay of 150,
    prove that X's pay as a percentage of Y's pay is 120%. -/
theorem employee_pay_percentage (total_pay : ℝ) (y_pay : ℝ) :
  total_pay = 330 →
  y_pay = 150 →
  (total_pay - y_pay) / y_pay * 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l2960_296018


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l2960_296049

open Real

noncomputable def f (x : ℝ) := exp x + exp (-x)

theorem tangent_line_y_intercept :
  let x₀ : ℝ := log (sqrt 2)
  let f' : ℝ → ℝ := λ x => exp x - exp (-x)
  let m : ℝ := f' x₀
  let b : ℝ := f x₀ - m * x₀
  (∀ x, f' (-x) = -f' x) →  -- f' is an odd function
  (m * (sqrt 2) / 2 = -1) →  -- tangent line is perpendicular to √2x + y + 1 = 0
  b = 3 * sqrt 2 / 2 - sqrt 2 / 4 * log 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l2960_296049


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2960_296081

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (sum_eq : p + q + r + s + t + u = 8) : 
  1/p + 9/q + 16/r + 25/s + 36/t + 49/u ≥ 84.5 ∧ 
  ∃ (p' q' r' s' t' u' : ℝ),
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧
    p' + q' + r' + s' + t' + u' = 8 ∧
    1/p' + 9/q' + 16/r' + 25/s' + 36/t' + 49/u' = 84.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2960_296081


namespace NUMINAMATH_CALUDE_intersection_area_formula_l2960_296027

/-- Regular octahedron with side length s -/
structure RegularOctahedron where
  s : ℝ
  s_pos : 0 < s

/-- Plane parallel to two opposite faces of the octahedron -/
structure ParallelPlane where
  distance_ratio : ℝ
  is_one_third : distance_ratio = 1/3

/-- The intersection of the plane and the octahedron forms a polygon -/
def intersection_polygon (o : RegularOctahedron) (p : ParallelPlane) : Set (ℝ × ℝ) := sorry

/-- The area of the intersection polygon -/
def intersection_area (o : RegularOctahedron) (p : ParallelPlane) : ℝ := sorry

/-- Theorem: The area of the intersection polygon is √3 * s^2 / 6 -/
theorem intersection_area_formula (o : RegularOctahedron) (p : ParallelPlane) :
  intersection_area o p = (Real.sqrt 3 * o.s^2) / 6 := by sorry

end NUMINAMATH_CALUDE_intersection_area_formula_l2960_296027


namespace NUMINAMATH_CALUDE_proportion_condition_l2960_296085

theorem proportion_condition (a b c : ℝ) (h : b ≠ 0 ∧ c ≠ 0) : 
  (∃ x y : ℝ, x / y = a / b ∧ y / x = b / c ∧ x^2 ≠ y * x) ∧
  (a / b = b / c → b^2 = a * c) ∧
  ¬(b^2 = a * c → a / b = b / c) :=
sorry

end NUMINAMATH_CALUDE_proportion_condition_l2960_296085


namespace NUMINAMATH_CALUDE_largest_square_leftover_l2960_296032

def yarn_length : ℕ := 35

theorem largest_square_leftover (s : ℕ) : 
  (s * 4 ≤ yarn_length) ∧ 
  (∀ t : ℕ, t * 4 ≤ yarn_length → t ≤ s) →
  yarn_length - s * 4 = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_square_leftover_l2960_296032


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2960_296014

theorem circle_diameter_from_area (A : Real) (r : Real) (d : Real) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2960_296014


namespace NUMINAMATH_CALUDE_rational_solutions_k_l2960_296042

/-- A function that checks if a given positive integer k results in rational solutions for the equation kx^2 + 20x + k = 0 -/
def has_rational_solutions (k : ℕ+) : Prop :=
  ∃ n : ℕ, (100 - k.val^2 : ℤ) = n^2

/-- The theorem stating that the positive integer values of k for which kx^2 + 20x + k = 0 has rational solutions are exactly 6, 8, and 10 -/
theorem rational_solutions_k :
  ∀ k : ℕ+, has_rational_solutions k ↔ k.val ∈ ({6, 8, 10} : Set ℕ) := by sorry

end NUMINAMATH_CALUDE_rational_solutions_k_l2960_296042


namespace NUMINAMATH_CALUDE_f_inequality_l2960_296012

-- Define the function f
variable (f : ℝ → ℝ)

-- State the condition f'(x) > f(x) for all x ∈ ℝ
variable (h : ∀ x : ℝ, (deriv f) x > f x)

-- Theorem statement
theorem f_inequality : f 2 > Real.exp 2 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2960_296012


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l2960_296068

def base5ToBase10 (n : List Nat) : Nat :=
  List.foldl (fun acc d => acc * 5 + d) 0 (List.reverse n)

theorem pirate_loot_sum :
  let silver := base5ToBase10 [1, 4, 3, 2]
  let spices := base5ToBase10 [2, 1, 3, 4]
  let silk := base5ToBase10 [3, 0, 2, 1]
  let books := base5ToBase10 [2, 3, 1]
  silver + spices + silk + books = 988 := by
  sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_l2960_296068


namespace NUMINAMATH_CALUDE_min_value_triangle_sides_l2960_296090

theorem min_value_triangle_sides (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 9) 
  (htri : x + y > z ∧ y + z > x ∧ z + x > y) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_triangle_sides_l2960_296090


namespace NUMINAMATH_CALUDE_remainder_3_pow_2024_mod_17_l2960_296098

theorem remainder_3_pow_2024_mod_17 : 3^2024 % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2024_mod_17_l2960_296098


namespace NUMINAMATH_CALUDE_matrix_product_abc_l2960_296038

def A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 3, -1; 0, 5, -4; -2, 5, 2]
def B : Matrix (Fin 3) (Fin 3) ℝ := !![3, -3, 0; 2, 1, -4; 5, 0, 1]
def C : Matrix (Fin 3) (Fin 2) ℝ := !![1, -1; 0, 2; 1, 0]

theorem matrix_product_abc :
  A * B * C = !![(-6 : ℝ), -13; -34, 20; -4, 8] := by sorry

end NUMINAMATH_CALUDE_matrix_product_abc_l2960_296038


namespace NUMINAMATH_CALUDE_sequence_general_term_l2960_296069

theorem sequence_general_term (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - 2 * a n = 2^n) →
  ∀ n : ℕ, n ≥ 1 → a n = n * 2^(n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2960_296069


namespace NUMINAMATH_CALUDE_extra_charge_per_wand_l2960_296045

def total_wands_bought : ℕ := 3
def cost_per_wand : ℚ := 60
def wands_sold : ℕ := 2
def total_collected : ℚ := 130

theorem extra_charge_per_wand :
  (total_collected / wands_sold) - cost_per_wand = 5 :=
by sorry

end NUMINAMATH_CALUDE_extra_charge_per_wand_l2960_296045


namespace NUMINAMATH_CALUDE_sqrt_equation_equivalence_l2960_296086

theorem sqrt_equation_equivalence (x : ℝ) (h : x > 6) :
  Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 6)) - 3 ↔ x ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_equivalence_l2960_296086


namespace NUMINAMATH_CALUDE_existence_of_solution_l2960_296029

theorem existence_of_solution (n : ℕ) (hn : n > 0) :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
  x^(n-1) + y^n = z^(n+1) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_solution_l2960_296029


namespace NUMINAMATH_CALUDE_simplify_fraction_l2960_296062

theorem simplify_fraction (a : ℝ) (h : a = 2) : (15 * a^4) / (75 * a^3) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2960_296062


namespace NUMINAMATH_CALUDE_correct_distribution_l2960_296034

/-- Represents the expenditure of a person -/
structure Expenditure where
  amount : ℚ

/-- Represents a person with their expenditure -/
structure Person where
  name : String
  expenditure : Expenditure

/-- Calculates the amount received by a person from the distribution -/
def amountReceived (p : Person) (total : ℚ) (sum : ℚ) : ℚ :=
  (p.expenditure.amount / sum) * total

theorem correct_distribution (a b c : Person)
  (h1 : b.expenditure.amount = 12/13 * a.expenditure.amount)
  (h2 : c.expenditure.amount = 2/3 * b.expenditure.amount)
  (h3 : a.name = "A" ∧ b.name = "B" ∧ c.name = "C") :
  let sum := a.expenditure.amount + b.expenditure.amount + c.expenditure.amount
  amountReceived a 9 sum = 6 ∧ amountReceived b 9 sum = 3 := by
  sorry

#check correct_distribution

end NUMINAMATH_CALUDE_correct_distribution_l2960_296034


namespace NUMINAMATH_CALUDE_no_integer_fourth_root_l2960_296074

theorem no_integer_fourth_root : ¬∃ (n : ℕ), n > 0 ∧ 5^4 + 12^4 + 9^4 + 8^4 = n^4 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_fourth_root_l2960_296074


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l2960_296036

theorem consecutive_integers_around_sqrt3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l2960_296036


namespace NUMINAMATH_CALUDE_hotel_nights_calculation_l2960_296048

theorem hotel_nights_calculation (total_value car_value hotel_cost_per_night : ℕ) 
  (h1 : total_value = 158000)
  (h2 : car_value = 30000)
  (h3 : hotel_cost_per_night = 4000) :
  (total_value - (car_value + 4 * car_value)) / hotel_cost_per_night = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotel_nights_calculation_l2960_296048


namespace NUMINAMATH_CALUDE_cash_percentage_proof_l2960_296087

/-- Calculates the percentage of total amount spent as cash given the total amount and amounts spent on raw materials and machinery. -/
def percentage_spent_as_cash (total_amount raw_materials machinery : ℚ) : ℚ :=
  ((total_amount - (raw_materials + machinery)) / total_amount) * 100

/-- Proves that given a total amount of $250, with $100 spent on raw materials and $125 spent on machinery, the percentage of the total amount spent as cash is 10%. -/
theorem cash_percentage_proof :
  percentage_spent_as_cash 250 100 125 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cash_percentage_proof_l2960_296087


namespace NUMINAMATH_CALUDE_unit_vector_AB_l2960_296093

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

theorem unit_vector_AB : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let magnitude := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let unit_vector := (AB.1 / magnitude, AB.2 / magnitude)
  unit_vector = (3/5, -4/5) :=
by sorry

end NUMINAMATH_CALUDE_unit_vector_AB_l2960_296093


namespace NUMINAMATH_CALUDE_dice_probability_l2960_296050

def num_dice : ℕ := 6
def sides_per_die : ℕ := 8

def probability_at_least_two_same : ℚ := 3781 / 4096

theorem dice_probability :
  probability_at_least_two_same = 1 - (sides_per_die.factorial / (sides_per_die - num_dice).factorial) / sides_per_die ^ num_dice :=
by sorry

end NUMINAMATH_CALUDE_dice_probability_l2960_296050


namespace NUMINAMATH_CALUDE_unique_function_property_l2960_296033

def iterateFunc (f : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, x => x
| (n + 1), x => f (iterateFunc f n x)

theorem unique_function_property (f : ℕ → ℕ) 
  (h : ∀ x y : ℕ, 0 ≤ y + f x - iterateFunc f (f y) x ∧ y + f x - iterateFunc f (f y) x ≤ 1) :
  ∀ n : ℕ, f n = n + 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_function_property_l2960_296033


namespace NUMINAMATH_CALUDE_olivers_bags_weight_l2960_296022

theorem olivers_bags_weight (james_bag_weight : ℝ) (oliver_bag_ratio : ℝ) : 
  james_bag_weight = 18 →
  oliver_bag_ratio = 1 / 6 →
  2 * (oliver_bag_ratio * james_bag_weight) = 6 := by
  sorry

end NUMINAMATH_CALUDE_olivers_bags_weight_l2960_296022


namespace NUMINAMATH_CALUDE_black_region_area_l2960_296064

/-- The area of the black region in a square-within-square configuration -/
theorem black_region_area (larger_side smaller_side : ℝ) (h1 : larger_side = 9) (h2 : smaller_side = 4) :
  larger_side ^ 2 - smaller_side ^ 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_black_region_area_l2960_296064


namespace NUMINAMATH_CALUDE_arrangements_with_constraints_l2960_296053

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def adjacent_arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

def doubly_adjacent_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 2)

theorem arrangements_with_constraints (n : ℕ) (h : n = 5) : 
  total_arrangements n - 2 * adjacent_arrangements n + doubly_adjacent_arrangements n = 36 := by
  sorry

#check arrangements_with_constraints

end NUMINAMATH_CALUDE_arrangements_with_constraints_l2960_296053


namespace NUMINAMATH_CALUDE_mixture_volume_l2960_296011

/-- Proves that the total volume of a mixture of two liquids is 4 liters -/
theorem mixture_volume (weight_a weight_b : ℝ) (ratio_a ratio_b : ℕ) (total_weight : ℝ) :
  weight_a = 950 →
  weight_b = 850 →
  ratio_a = 3 →
  ratio_b = 2 →
  total_weight = 3640 →
  ∃ (vol_a vol_b : ℝ),
    vol_a / vol_b = ratio_a / ratio_b ∧
    total_weight = vol_a * weight_a + vol_b * weight_b ∧
    vol_a + vol_b = 4 :=
by sorry

end NUMINAMATH_CALUDE_mixture_volume_l2960_296011


namespace NUMINAMATH_CALUDE_differential_of_y_l2960_296063

noncomputable def y (x : ℝ) : ℝ := 2 * x + Real.log (|Real.sin x + 2 * Real.cos x|)

theorem differential_of_y (x : ℝ) :
  deriv y x = (5 * Real.cos x) / (Real.sin x + 2 * Real.cos x) :=
by sorry

end NUMINAMATH_CALUDE_differential_of_y_l2960_296063


namespace NUMINAMATH_CALUDE_intersection_segment_equals_incircle_diameter_l2960_296016

/-- Right triangle with incircle and two circles on hypotenuse endpoints -/
structure RightTriangleWithCircles where
  -- Legs of the right triangle
  a : ℝ
  b : ℝ
  -- Hypotenuse of the right triangle
  c : ℝ
  -- Radius of the incircle
  r : ℝ
  -- The triangle is right-angled
  right_angle : a^2 + b^2 = c^2
  -- The incircle exists and touches all sides
  incircle : a + b - c = 2 * r
  -- All lengths are positive
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  r_pos : r > 0

/-- The length of the intersection segment equals the incircle diameter -/
theorem intersection_segment_equals_incircle_diameter 
  (t : RightTriangleWithCircles) : a + b - c = 2 * r :=
by sorry

end NUMINAMATH_CALUDE_intersection_segment_equals_incircle_diameter_l2960_296016


namespace NUMINAMATH_CALUDE_circle_radius_is_zero_l2960_296083

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 10*y + 41 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 0

/-- Theorem: The radius of the circle described by the given equation is 0 -/
theorem circle_radius_is_zero :
  ∀ x y : ℝ, circle_equation x y → ∃ c : ℝ × ℝ, ∀ p : ℝ × ℝ, circle_equation p.1 p.2 ↔ (p.1 - c.1)^2 + (p.2 - c.2)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_zero_l2960_296083
