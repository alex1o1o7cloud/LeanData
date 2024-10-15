import Mathlib

namespace NUMINAMATH_CALUDE_similar_triangles_collinearity_l2131_213133

/-- Two triangles are similar if they have the same shape but possibly different size and orientation -/
def similar_triangles (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : Prop := sorry

/-- Two triangles are differently oriented if they cannot be made to coincide by translation and scaling -/
def differently_oriented (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : Prop := sorry

/-- A point divides a segment in a given ratio -/
def divides_segment_in_ratio (A A' A₁ : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ t : ℝ, A' = (1 - t) • A + t • A₁ ∧ r = t / (1 - t)

/-- Three points are collinear if they lie on a single straight line -/
def collinear (A B C : ℝ × ℝ) : Prop := sorry

theorem similar_triangles_collinearity 
  (A B C A₁ B₁ C₁ A' B' C' : ℝ × ℝ) 
  (ABC : Set (Fin 3 → ℝ × ℝ)) 
  (A₁B₁C₁ : Set (Fin 3 → ℝ × ℝ)) 
  (h_similar : similar_triangles ABC A₁B₁C₁)
  (h_oriented : differently_oriented ABC A₁B₁C₁)
  (h_A' : divides_segment_in_ratio A A' A₁ (dist B C / dist B₁ C₁))
  (h_B' : divides_segment_in_ratio B B' B₁ (dist B C / dist B₁ C₁))
  (h_C' : divides_segment_in_ratio C C' C₁ (dist B C / dist B₁ C₁)) :
  collinear A' B' C' := by sorry

end NUMINAMATH_CALUDE_similar_triangles_collinearity_l2131_213133


namespace NUMINAMATH_CALUDE_recurring_decimal_product_l2131_213100

theorem recurring_decimal_product : 
  (8 : ℚ) / 99 * (4 : ℚ) / 11 = (32 : ℚ) / 1089 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_product_l2131_213100


namespace NUMINAMATH_CALUDE_numbers_solution_l2131_213176

def find_numbers (x y z : ℤ) : Prop :=
  (y = 2*x - 3) ∧ 
  (x + y = 51) ∧ 
  (z = 4*x - y)

theorem numbers_solution : 
  ∃ (x y z : ℤ), find_numbers x y z ∧ x = 18 ∧ y = 33 ∧ z = 39 :=
sorry

end NUMINAMATH_CALUDE_numbers_solution_l2131_213176


namespace NUMINAMATH_CALUDE_exam_score_problem_l2131_213166

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 130 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 38 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l2131_213166


namespace NUMINAMATH_CALUDE_binomial_sum_expectation_variance_l2131_213162

/-- A random variable X following a binomial distribution B(n, p) -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  X : ℝ

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV n p) : ℝ := n * p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV n p) : ℝ := n * p * (1 - p)

theorem binomial_sum_expectation_variance
  (X : BinomialRV 10 0.6) (Y : ℝ) 
  (h₁ : X.X + Y = 10) :
  expectation X = 6 ∧ 
  variance X = 2.4 ∧ 
  expectation X + Y = 10 → 
  Y = 4 ∧ variance X = 2.4 :=
sorry

end NUMINAMATH_CALUDE_binomial_sum_expectation_variance_l2131_213162


namespace NUMINAMATH_CALUDE_max_distance_circle_C_to_line_L_l2131_213139

/-- Circle C with equation x^2 + y^2 - 4x + m = 0 -/
def circle_C (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 + m = 0}

/-- Circle D with equation (x-3)^2 + (y+2√2)^2 = 4 -/
def circle_D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1-3)^2 + (p.2+2*Real.sqrt 2)^2 = 4}

/-- Line L with equation 3x - 4y + 4 = 0 -/
def line_L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3*p.1 - 4*p.2 + 4 = 0}

/-- The distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (L : Set (ℝ × ℝ)) : ℝ := sorry

/-- The maximum distance from any point on a set to a line -/
def max_distance_set_to_line (S : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) : ℝ := sorry

theorem max_distance_circle_C_to_line_L :
  ∃ m : ℝ,
    (circle_C m).Nonempty ∧
    (∃ p : ℝ × ℝ, p ∈ circle_C m ∧ p ∈ circle_D) ∧
    max_distance_set_to_line (circle_C m) line_L = 3 := by sorry

end NUMINAMATH_CALUDE_max_distance_circle_C_to_line_L_l2131_213139


namespace NUMINAMATH_CALUDE_cubic_is_odd_rhombus_diagonals_bisect_l2131_213179

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of diagonals bisecting each other
def diagonals_bisect (shape : Type) : Prop := 
  ∀ d1 d2 : shape → ℝ × ℝ, d1 ≠ d2 → ∃ p : ℝ × ℝ, 
    (∃ a b : shape, d1 a = p ∧ d2 b = p) ∧
    (∀ q : shape, d1 q = p ∨ d2 q = p)

-- Define shapes
class Parallelogram (shape : Type)
class Rhombus (shape : Type) extends Parallelogram shape

-- Theorem statements
theorem cubic_is_odd : is_odd_function f := sorry

theorem rhombus_diagonals_bisect (shape : Type) [Rhombus shape] : 
  diagonals_bisect shape := sorry

end NUMINAMATH_CALUDE_cubic_is_odd_rhombus_diagonals_bisect_l2131_213179


namespace NUMINAMATH_CALUDE_age_difference_l2131_213153

/-- Proves that the age difference between a man and his son is 26 years, given the specified conditions. -/
theorem age_difference (son_age man_age : ℕ) : 
  son_age = 24 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 26 := by
  sorry

#check age_difference

end NUMINAMATH_CALUDE_age_difference_l2131_213153


namespace NUMINAMATH_CALUDE_abs_negative_2023_l2131_213110

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l2131_213110


namespace NUMINAMATH_CALUDE_lily_newspaper_collection_l2131_213188

/-- Given that Chris collected 42 newspapers and the total number of newspapers
    collected by Chris and Lily is 65, prove that Lily collected 23 newspapers. -/
theorem lily_newspaper_collection (chris_newspapers lily_newspapers total_newspapers : ℕ) :
  chris_newspapers = 42 →
  total_newspapers = 65 →
  total_newspapers = chris_newspapers + lily_newspapers →
  lily_newspapers = 23 := by
sorry

end NUMINAMATH_CALUDE_lily_newspaper_collection_l2131_213188


namespace NUMINAMATH_CALUDE_linear_function_condition_l2131_213191

/-- Given a linear function f(x) = ax - x - a where a > 0 and a ≠ 1, 
    prove that a > 1. -/
theorem linear_function_condition (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) : a > 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_condition_l2131_213191


namespace NUMINAMATH_CALUDE_euclid_wrote_elements_l2131_213146

/-- The author of "Elements" -/
def author_of_elements : String := "Euclid"

/-- Theorem stating that Euclid is the author of "Elements" -/
theorem euclid_wrote_elements : author_of_elements = "Euclid" := by sorry

end NUMINAMATH_CALUDE_euclid_wrote_elements_l2131_213146


namespace NUMINAMATH_CALUDE_fraction_simplification_l2131_213194

theorem fraction_simplification : 
  (45 : ℚ) / 28 * 49 / 75 * 100 / 63 = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2131_213194


namespace NUMINAMATH_CALUDE_conference_games_l2131_213195

/-- Calculates the number of games within a division -/
def games_within_division (n : ℕ) : ℕ := n * (n - 1)

/-- Calculates the number of games between two divisions -/
def games_between_divisions (n m : ℕ) : ℕ := 2 * n * m

/-- The total number of games in the conference -/
def total_games : ℕ :=
  let div_a := 6
  let div_b := 7
  let div_c := 5
  let within_a := games_within_division div_a
  let within_b := games_within_division div_b
  let within_c := games_within_division div_c
  let between_ab := games_between_divisions div_a div_b
  let between_ac := games_between_divisions div_a div_c
  let between_bc := games_between_divisions div_b div_c
  within_a + within_b + within_c + between_ab + between_ac + between_bc

theorem conference_games : total_games = 306 := by sorry

end NUMINAMATH_CALUDE_conference_games_l2131_213195


namespace NUMINAMATH_CALUDE_g_x_minus_3_l2131_213112

/-- The function g(x) = x^2 -/
def g (x : ℝ) : ℝ := x^2

/-- Theorem: For the function g(x) = x^2, g(x-3) = x^2 - 6x + 9 -/
theorem g_x_minus_3 (x : ℝ) : g (x - 3) = x^2 - 6*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_g_x_minus_3_l2131_213112


namespace NUMINAMATH_CALUDE_polynomial_coefficient_equality_l2131_213136

theorem polynomial_coefficient_equality (m n : ℤ) : 
  (∀ x : ℝ, (x - 1) * (x + m) = x^2 - n*x - 6) → 
  (m = 6 ∧ n = -5) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_equality_l2131_213136


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_sum_l2131_213135

theorem geometric_sequence_minimum_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  a 6 = 3 →  -- a₆ = 3
  ∃ m : ℝ, m = 6 ∧ ∀ q, q > 0 → a 4 + a 8 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_sum_l2131_213135


namespace NUMINAMATH_CALUDE_complement_of_A_l2131_213197

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x > 1}

-- State the theorem
theorem complement_of_A : 
  Set.compl A = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2131_213197


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l2131_213160

/-- An isosceles right triangle with given area and hypotenuse length -/
structure IsoscelesRightTriangle where
  -- The length of a leg
  leg : ℝ
  -- The area of the triangle
  area : ℝ
  -- The condition that the area is equal to half the square of the leg
  area_eq : area = leg^2 / 2

/-- The theorem stating that an isosceles right triangle with area 9 has hypotenuse length 6 -/
theorem isosceles_right_triangle_hypotenuse (t : IsoscelesRightTriangle) 
  (h_area : t.area = 9) : 
  t.leg * Real.sqrt 2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l2131_213160


namespace NUMINAMATH_CALUDE_pen_distribution_l2131_213193

theorem pen_distribution (num_pencils : ℕ) (num_students : ℕ) (num_pens : ℕ) : 
  num_pencils = 910 →
  num_students = 91 →
  num_pencils % num_students = 0 →
  num_pens % num_students = 0 →
  ∃ k : ℕ, num_pens = k * num_students :=
by sorry

end NUMINAMATH_CALUDE_pen_distribution_l2131_213193


namespace NUMINAMATH_CALUDE_michael_sarah_games_l2131_213177

def total_players : ℕ := 12
def players_per_game : ℕ := 6

theorem michael_sarah_games (michael sarah : Fin total_players) 
  (h_distinct : michael ≠ sarah) :
  (Finset.univ.filter (λ game : Finset (Fin total_players) => 
    game.card = players_per_game ∧ 
    michael ∈ game ∧ 
    sarah ∈ game)).card = Nat.choose (total_players - 2) (players_per_game - 2) := by
  sorry

end NUMINAMATH_CALUDE_michael_sarah_games_l2131_213177


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_exists_l2131_213123

theorem quadratic_equation_solution_exists (a b c : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * b * x + c = 0) ∨ 
  (∃ x : ℝ, b * x^2 + 2 * c * x + a = 0) ∨ 
  (∃ x : ℝ, c * x^2 + 2 * a * x + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_exists_l2131_213123


namespace NUMINAMATH_CALUDE_xy_value_l2131_213119

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2131_213119


namespace NUMINAMATH_CALUDE_remainder_2468135792_mod_101_l2131_213118

theorem remainder_2468135792_mod_101 : 2468135792 % 101 = 52 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2468135792_mod_101_l2131_213118


namespace NUMINAMATH_CALUDE_least_valid_number_l2131_213154

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧  -- Four-digit positive integer
  ∃ (a b c d : ℕ), 
    n = 1000 * a + 100 * b + 10 * c + d ∧  -- Digit representation
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- All digits are different
    (a = 5 ∨ b = 5 ∨ c = 5 ∨ d = 5) ∧  -- One of the digits is 5
    n % a = 0 ∧ n % b = 0 ∧ n % c = 0 ∧ n % d = 0  -- Divisible by each of its digits

theorem least_valid_number : 
  is_valid_number 1524 ∧ ∀ m : ℕ, is_valid_number m → m ≥ 1524 :=
sorry

end NUMINAMATH_CALUDE_least_valid_number_l2131_213154


namespace NUMINAMATH_CALUDE_modulo_nine_sum_l2131_213150

theorem modulo_nine_sum (n : ℕ) : n = 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 → 0 ≤ n ∧ n < 9 → n % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_modulo_nine_sum_l2131_213150


namespace NUMINAMATH_CALUDE_more_I_than_P_l2131_213121

/-- Sum of digits of a natural number -/
def S (n : ℕ) : ℕ := sorry

/-- Property P: all terms in the sequence n, S(n), S(S(n)),... are even -/
def has_property_P (n : ℕ) : Prop := sorry

/-- Property I: all terms in the sequence n, S(n), S(S(n)),... are odd -/
def has_property_I (n : ℕ) : Prop := sorry

/-- Count of numbers with property P in the range 1 to 2017 -/
def count_P : ℕ := sorry

/-- Count of numbers with property I in the range 1 to 2017 -/
def count_I : ℕ := sorry

theorem more_I_than_P : count_I > count_P := by sorry

end NUMINAMATH_CALUDE_more_I_than_P_l2131_213121


namespace NUMINAMATH_CALUDE_triangle_area_l2131_213184

/-- Given a triangle with sides 6, 8, and 10, its area is 24 -/
theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2131_213184


namespace NUMINAMATH_CALUDE_managers_salary_l2131_213120

def employee_count : ℕ := 50
def initial_average_salary : ℚ := 2500
def average_increase : ℚ := 150

theorem managers_salary (manager_salary : ℚ) :
  (employee_count * initial_average_salary + manager_salary) / (employee_count + 1) =
  initial_average_salary + average_increase →
  manager_salary = 10150 := by
sorry

end NUMINAMATH_CALUDE_managers_salary_l2131_213120


namespace NUMINAMATH_CALUDE_not_monotone_decreasing_periodic_function_l2131_213169

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Theorem 1: If f(1) > f(-1), then f is not monotonically decreasing on ℝ
theorem not_monotone_decreasing (h : f 1 > f (-1)) : 
  ¬ (∀ x y : ℝ, x ≤ y → f x ≥ f y) := by sorry

-- Theorem 2: If f(1+x) = f(x-1) for all x ∈ ℝ, then f is periodic
theorem periodic_function (h : ∀ x : ℝ, f (1 + x) = f (x - 1)) : 
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x := by sorry

end NUMINAMATH_CALUDE_not_monotone_decreasing_periodic_function_l2131_213169


namespace NUMINAMATH_CALUDE_rational_coordinates_solution_l2131_213101

theorem rational_coordinates_solution (x : ℚ) : ∃ y : ℚ, 2 * x^3 + 2 * y^3 - 3 * x^2 - 3 * y^2 + 1 = 0 := by
  -- We claim that y = 1 - x satisfies the equation
  let y := 1 - x
  -- Existential introduction
  use y
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_rational_coordinates_solution_l2131_213101


namespace NUMINAMATH_CALUDE_hay_from_grass_l2131_213186

/-- The amount of hay obtained from freshly cut grass -/
theorem hay_from_grass (initial_mass : ℝ) (grass_moisture : ℝ) (hay_moisture : ℝ) : 
  initial_mass = 1000 →
  grass_moisture = 0.6 →
  hay_moisture = 0.15 →
  (initial_mass * (1 - grass_moisture)) / (1 - hay_moisture) = 470^10 / 17 := by
  sorry

#eval (470^10 : ℚ) / 17

end NUMINAMATH_CALUDE_hay_from_grass_l2131_213186


namespace NUMINAMATH_CALUDE_s_equality_l2131_213127

theorem s_equality (x : ℝ) : 
  (x - 2)^4 + 4*(x - 2)^3 + 6*(x - 2)^2 + 4*(x - 2) + 1 = (x - 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_s_equality_l2131_213127


namespace NUMINAMATH_CALUDE_injury_healing_ratio_l2131_213152

/-- The number of days it takes for the pain to subside -/
def pain_subsided : ℕ := 3

/-- The number of days James waits after full healing before working out -/
def wait_before_workout : ℕ := 3

/-- The number of days James waits before lifting heavy -/
def wait_before_heavy : ℕ := 21

/-- The total number of days until James can lift heavy again -/
def total_days : ℕ := 39

/-- The number of days it takes for the injury to fully heal -/
def healing_time : ℕ := total_days - pain_subsided - wait_before_workout - wait_before_heavy

/-- The ratio of healing time to pain subsided time -/
def healing_ratio : ℚ := healing_time / pain_subsided

theorem injury_healing_ratio : healing_ratio = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_injury_healing_ratio_l2131_213152


namespace NUMINAMATH_CALUDE_one_fifths_in_ten_thirds_l2131_213122

theorem one_fifths_in_ten_thirds :
  (10 : ℚ) / 3 / (1 / 5) = 50 / 3 := by sorry

end NUMINAMATH_CALUDE_one_fifths_in_ten_thirds_l2131_213122


namespace NUMINAMATH_CALUDE_tan_105_degrees_l2131_213137

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l2131_213137


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_no_real_roots_equation_four_solutions_l2131_213167

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  (x + 1)^2 - 5 = 0 ↔ x = Real.sqrt 5 - 1 ∨ x = -Real.sqrt 5 - 1 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  2 * x^2 - 4 * x + 1 = 0 ↔ x = (4 + Real.sqrt 8) / 4 ∨ x = (4 - Real.sqrt 8) / 4 := by sorry

-- Equation 3
theorem equation_three_no_real_roots :
  ¬∃ (x : ℝ), (2 * x + 1) * (x - 3) = -7 := by sorry

-- Equation 4
theorem equation_four_solutions (x : ℝ) :
  3 * (x - 2)^2 = x * (x - 2) ↔ x = 2 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_no_real_roots_equation_four_solutions_l2131_213167


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2131_213129

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π / 4 - α) = -4 / 5) : 
  Real.sin (2 * α) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2131_213129


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l2131_213117

theorem distance_from_origin_to_point : ∀ (x y : ℝ), 
  x = 12 ∧ y = 9 → Real.sqrt (x^2 + y^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l2131_213117


namespace NUMINAMATH_CALUDE_actual_payment_calculation_l2131_213116

/-- Represents the restaurant's voucher system and discount policy -/
structure Restaurant where
  voucher_cost : ℕ := 25
  voucher_value : ℕ := 50
  max_vouchers : ℕ := 3
  hotpot_base_cost : ℕ := 50
  other_dishes_discount : ℚ := 0.4

/-- Represents a family's dining experience -/
structure DiningExperience where
  restaurant : Restaurant
  total_bill : ℕ
  voucher_savings : ℕ
  onsite_discount_savings : ℕ

/-- The theorem to be proved -/
theorem actual_payment_calculation (d : DiningExperience) :
  d.restaurant.hotpot_base_cost = 50 ∧
  d.restaurant.voucher_cost = 25 ∧
  d.restaurant.voucher_value = 50 ∧
  d.restaurant.max_vouchers = 3 ∧
  d.restaurant.other_dishes_discount = 0.4 ∧
  d.onsite_discount_savings = d.voucher_savings + 15 →
  d.total_bill - d.onsite_discount_savings = 185 := by
  sorry


end NUMINAMATH_CALUDE_actual_payment_calculation_l2131_213116


namespace NUMINAMATH_CALUDE_interval_length_theorem_l2131_213159

theorem interval_length_theorem (c d : ℝ) : 
  (∃ (x : ℝ), c ≤ 3 * x + 5 ∧ 3 * x + 5 ≤ d) → 
  ((d - 5) / 3 - (c - 5) / 3 = 15) → 
  d - c = 45 := by
sorry

end NUMINAMATH_CALUDE_interval_length_theorem_l2131_213159


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_area_ratio_l2131_213113

/-- The ratio of the perimeter to the area of an equilateral triangle with side length 8 units is √3/2 -/
theorem equilateral_triangle_perimeter_area_ratio :
  let s : ℝ := 8
  let perimeter : ℝ := 3 * s
  let area : ℝ := s^2 * Real.sqrt 3 / 4
  perimeter / area = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_area_ratio_l2131_213113


namespace NUMINAMATH_CALUDE_river_boat_journey_time_l2131_213102

theorem river_boat_journey_time 
  (river_speed : ℝ) 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (h1 : river_speed = 2) 
  (h2 : boat_speed = 6) 
  (h3 : distance = 32) : 
  (distance / (boat_speed - river_speed) + distance / (boat_speed + river_speed)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_river_boat_journey_time_l2131_213102


namespace NUMINAMATH_CALUDE_savings_multiple_l2131_213172

theorem savings_multiple (monthly_pay : ℝ) (savings_fraction : ℝ) : 
  savings_fraction = 0.29411764705882354 →
  monthly_pay > 0 →
  let monthly_savings := monthly_pay * savings_fraction
  let monthly_non_savings := monthly_pay - monthly_savings
  let total_savings := monthly_savings * 12
  total_savings = 5 * monthly_non_savings :=
by sorry

end NUMINAMATH_CALUDE_savings_multiple_l2131_213172


namespace NUMINAMATH_CALUDE_correct_division_formula_l2131_213107

theorem correct_division_formula : 
  (240 : ℕ) / (13 + 11) = 240 / (13 + 11) := by sorry

end NUMINAMATH_CALUDE_correct_division_formula_l2131_213107


namespace NUMINAMATH_CALUDE_yellow_preference_l2131_213198

theorem yellow_preference (total_students : ℕ) (total_girls : ℕ) 
  (h_total : total_students = 30)
  (h_girls : total_girls = 18)
  (h_green : total_students / 2 = total_students - (total_students / 2))
  (h_pink : total_girls / 3 = total_girls - (2 * (total_girls / 3))) :
  total_students - (total_students / 2 + total_girls / 3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_yellow_preference_l2131_213198


namespace NUMINAMATH_CALUDE_inequality_proof_l2131_213128

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_prod : a * b * c = 1) : 
  (a + 1/b)^2 + (b + 1/c)^2 + (c + 1/a)^2 ≥ 3 * (a + b + c + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2131_213128


namespace NUMINAMATH_CALUDE_problem_statement_l2131_213199

theorem problem_statement : ((18^10 / 18^9)^3 * 16^3) / 8^6 = 91.125 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2131_213199


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l2131_213132

theorem wire_cutting_problem (shorter_piece longer_piece total_length : ℝ) :
  shorter_piece = 40 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 140 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l2131_213132


namespace NUMINAMATH_CALUDE_trig_identity_l2131_213149

theorem trig_identity : 
  1 / Real.cos (70 * π / 180) + Real.sqrt 2 / Real.sin (70 * π / 180) = 
  4 * Real.sin (65 * π / 180) / Real.sin (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2131_213149


namespace NUMINAMATH_CALUDE_min_value_sum_product_l2131_213125

theorem min_value_sum_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 1 → a + 2 * b ≤ x + 2 * y ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 1 ∧ x + 2 * y = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l2131_213125


namespace NUMINAMATH_CALUDE_lewis_earnings_l2131_213156

theorem lewis_earnings (weeks : ℕ) (weekly_rent : ℚ) (total_after_rent : ℚ) 
  (h1 : weeks = 233)
  (h2 : weekly_rent = 49)
  (h3 : total_after_rent = 93899) :
  (total_after_rent + weeks * weekly_rent) / weeks = 451.99 := by
sorry

end NUMINAMATH_CALUDE_lewis_earnings_l2131_213156


namespace NUMINAMATH_CALUDE_circle_intersection_condition_l2131_213155

/-- Circle B with equation x^2 + y^2 + b = 0 -/
def circle_B (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + b = 0}

/-- Circle C with equation x^2 + y^2 - 6x + 8y + 16 = 0 -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 + 8*p.2 + 16 = 0}

/-- Two circles do not intersect -/
def no_intersection (A B : Set (ℝ × ℝ)) : Prop :=
  A ∩ B = ∅

/-- The main theorem -/
theorem circle_intersection_condition (b : ℝ) :
  no_intersection (circle_B b) circle_C →
  (-4 < b ∧ b < 0) ∨ b < -25 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_condition_l2131_213155


namespace NUMINAMATH_CALUDE_shielas_paint_colors_l2131_213114

theorem shielas_paint_colors (total_blocks : ℕ) (blocks_per_color : ℕ) (h1 : total_blocks = 196) (h2 : blocks_per_color = 14) : 
  total_blocks / blocks_per_color = 14 := by
  sorry

end NUMINAMATH_CALUDE_shielas_paint_colors_l2131_213114


namespace NUMINAMATH_CALUDE_max_n_geometric_sequence_l2131_213115

-- Define the geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem max_n_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  a 2 * a 4 = 4 →  -- a₂ · a₄ = 4
  a 1 + a 2 + a 3 = 14 →  -- a₁ + a₂ + a₃ = 14
  (∃ a₁ q, ∀ n, a n = geometric_sequence a₁ q n) →  -- {a_n} is a geometric sequence
  (∀ n > 4, a n * a (n+1) * a (n+2) ≤ 1/9) ∧  -- For all n > 4, the product is ≤ 1/9
  (a 4 * a 5 * a 6 > 1/9) :=  -- For n = 4, the product is > 1/9
by sorry

end NUMINAMATH_CALUDE_max_n_geometric_sequence_l2131_213115


namespace NUMINAMATH_CALUDE_garden_width_is_ten_l2131_213138

/-- Represents a rectangular garden with specific properties. -/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  perimeter_eq : width * 2 + length * 2 = 60
  area_eq : width * length = 200
  length_twice_width : length = 2 * width

/-- Theorem stating that a rectangular garden with the given properties has a width of 10 meters. -/
theorem garden_width_is_ten (garden : RectangularGarden) : garden.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_garden_width_is_ten_l2131_213138


namespace NUMINAMATH_CALUDE_fruit_basket_total_l2131_213180

/-- Calculates the total number of fruits in a basket given the number of oranges and relationships between fruit quantities. -/
def totalFruits (oranges : ℕ) : ℕ :=
  let apples := oranges - 2
  let bananas := 3 * apples
  let peaches := bananas / 2
  oranges + apples + bananas + peaches

/-- Theorem stating that the total number of fruits in the basket is 28. -/
theorem fruit_basket_total : totalFruits 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_total_l2131_213180


namespace NUMINAMATH_CALUDE_sin_double_angle_l2131_213173

theorem sin_double_angle (x : Real) (h : Real.sin (x - π/4) = 2/3) : 
  Real.sin (2*x) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_l2131_213173


namespace NUMINAMATH_CALUDE_negative_integer_sum_square_l2131_213190

theorem negative_integer_sum_square (N : ℤ) : 
  N < 0 → N^2 + N = -12 → N = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_sum_square_l2131_213190


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2131_213103

/-- Two vectors in R² are parallel if their coordinates are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, 
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (4, x + 1)
  parallel a b → x = 1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2131_213103


namespace NUMINAMATH_CALUDE_exactly_two_win_probability_l2131_213192

/-- The probability that exactly two out of three players win a game, given their individual probabilities of success. -/
theorem exactly_two_win_probability 
  (p_alice : ℚ) 
  (p_benjamin : ℚ) 
  (p_carol : ℚ) 
  (h_alice : p_alice = 1/5) 
  (h_benjamin : p_benjamin = 3/8) 
  (h_carol : p_carol = 2/7) : 
  (p_alice * p_benjamin * (1 - p_carol) + 
   p_alice * p_carol * (1 - p_benjamin) + 
   p_benjamin * p_carol * (1 - p_alice)) = 49/280 := by
sorry


end NUMINAMATH_CALUDE_exactly_two_win_probability_l2131_213192


namespace NUMINAMATH_CALUDE_library_book_count_l2131_213163

/-- The number of books the library had before the grant -/
def initial_books : ℕ := 5935

/-- The number of books purchased with the grant -/
def purchased_books : ℕ := 2647

/-- The total number of books after the grant -/
def total_books : ℕ := initial_books + purchased_books

theorem library_book_count : total_books = 8582 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l2131_213163


namespace NUMINAMATH_CALUDE_faster_speed_calculation_l2131_213170

/-- Proves that the faster speed is 20 km/hr given the conditions of the problem -/
theorem faster_speed_calculation (actual_distance : ℝ) (actual_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 20)
  (h2 : actual_speed = 10)
  (h3 : additional_distance = 20) :
  let time := actual_distance / actual_speed
  let total_distance := actual_distance + additional_distance
  let faster_speed := total_distance / time
  faster_speed = 20 := by sorry

end NUMINAMATH_CALUDE_faster_speed_calculation_l2131_213170


namespace NUMINAMATH_CALUDE_infinitely_many_special_prisms_l2131_213175

/-- A rectangular prism with two equal edges and the third differing by 1 -/
structure SpecialPrism where
  a : ℕ
  b : ℕ
  h_b : b = a + 1 ∨ b = a - 1

/-- The body diagonal of a rectangular prism is an integer -/
def has_integer_diagonal (p : SpecialPrism) : Prop :=
  ∃ d : ℕ, d^2 = 2 * p.a^2 + p.b^2

/-- There are infinitely many rectangular prisms with integer edges and diagonal,
    where two edges are equal and the third differs by 1 -/
theorem infinitely_many_special_prisms :
  ∀ n : ℕ, ∃ (prisms : Finset SpecialPrism),
    prisms.card > n ∧ ∀ p ∈ prisms, has_integer_diagonal p :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_special_prisms_l2131_213175


namespace NUMINAMATH_CALUDE_hot_dogs_leftover_l2131_213147

theorem hot_dogs_leftover : 36159782 % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_leftover_l2131_213147


namespace NUMINAMATH_CALUDE_sqrt_two_squared_inverse_half_l2131_213140

theorem sqrt_two_squared_inverse_half : 
  (((-Real.sqrt 2)^2)^(-1/2 : ℝ)) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_inverse_half_l2131_213140


namespace NUMINAMATH_CALUDE_sum_of_median_scores_l2131_213161

def median_score_A : ℕ := 28
def median_score_B : ℕ := 36

theorem sum_of_median_scores :
  median_score_A + median_score_B = 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_median_scores_l2131_213161


namespace NUMINAMATH_CALUDE_nancy_physical_education_marks_l2131_213164

def american_literature : ℕ := 66
def history : ℕ := 75
def home_economics : ℕ := 52
def art : ℕ := 89
def average_marks : ℕ := 70
def total_subjects : ℕ := 5

theorem nancy_physical_education_marks :
  let known_subjects_total := american_literature + history + home_economics + art
  let total_marks := average_marks * total_subjects
  total_marks - known_subjects_total = 68 := by
sorry

end NUMINAMATH_CALUDE_nancy_physical_education_marks_l2131_213164


namespace NUMINAMATH_CALUDE_chess_tournament_players_l2131_213109

theorem chess_tournament_players (total_games : ℕ) (h : total_games = 240) :
  ∃ n : ℕ, n > 0 ∧ 2 * n * (n - 1) = total_games ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l2131_213109


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2131_213157

open Real

theorem function_inequality_implies_a_range (a : ℝ) :
  (∀ x ≥ 0, 2 * (exp x) - 2 * a * x - a^2 + 3 - x^2 ≥ 0) →
  a ∈ Set.Icc (-Real.sqrt 5) (3 - Real.log 3) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2131_213157


namespace NUMINAMATH_CALUDE_inradius_less_than_half_side_height_bound_l2131_213104

/-- Triangle ABC with side lengths a, b, c, angles A, B, C, inradius r, circumradius R, and height h_a from vertex A to side BC -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  r : ℝ
  R : ℝ
  h_a : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The inradius is less than half of any side length -/
theorem inradius_less_than_half_side (t : Triangle) : 2 * t.r < t.a := by sorry

/-- The height to side a is at most twice the circumradius times the square of the cosine of half angle A -/
theorem height_bound (t : Triangle) : t.h_a ≤ 2 * t.R * (Real.cos (t.A / 2))^2 := by sorry

end NUMINAMATH_CALUDE_inradius_less_than_half_side_height_bound_l2131_213104


namespace NUMINAMATH_CALUDE_units_digit_153_base3_l2131_213158

/-- Converts a natural number to its base 3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The units digit is the last digit in the list representation -/
def unitsDigit (digits : List ℕ) : ℕ :=
  match digits.reverse with
  | [] => 0
  | d :: _ => d

theorem units_digit_153_base3 :
  unitsDigit (toBase3 153) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_153_base3_l2131_213158


namespace NUMINAMATH_CALUDE_river_trip_longer_than_lake_l2131_213174

/-- Proves that a round trip on a river takes longer than traveling the same distance on a lake -/
theorem river_trip_longer_than_lake (v w : ℝ) (h : v > w) (h_pos : v > 0) :
  (20 * v) / (v^2 - w^2) > 20 / v := by
  sorry

end NUMINAMATH_CALUDE_river_trip_longer_than_lake_l2131_213174


namespace NUMINAMATH_CALUDE_smallest_seem_l2131_213178

/-- Represents a digit mapping for the puzzle MY + ROZH = SEEM -/
structure DigitMapping where
  m : Nat
  y : Nat
  r : Nat
  o : Nat
  z : Nat
  h : Nat
  s : Nat
  e : Nat
  unique : m ≠ y ∧ m ≠ r ∧ m ≠ o ∧ m ≠ z ∧ m ≠ h ∧ m ≠ s ∧ m ≠ e ∧
           y ≠ r ∧ y ≠ o ∧ y ≠ z ∧ y ≠ h ∧ y ≠ s ∧ y ≠ e ∧
           r ≠ o ∧ r ≠ z ∧ r ≠ h ∧ r ≠ s ∧ r ≠ e ∧
           o ≠ z ∧ o ≠ h ∧ o ≠ s ∧ o ≠ e ∧
           z ≠ h ∧ z ≠ s ∧ z ≠ e ∧
           h ≠ s ∧ h ≠ e ∧
           s ≠ e
  valid_digits : m < 10 ∧ y < 10 ∧ r < 10 ∧ o < 10 ∧ z < 10 ∧ h < 10 ∧ s < 10 ∧ e < 10
  s_greater_than_one : s > 1

/-- The equation MY + ROZH = SEEM holds for the given digit mapping -/
def equation_holds (d : DigitMapping) : Prop :=
  10 * d.m + d.y + 1000 * d.r + 100 * d.o + 10 * d.z + d.h = 1000 * d.s + 100 * d.e + 10 * d.e + d.m

/-- There exists a valid digit mapping for which the equation holds -/
def exists_valid_mapping : Prop :=
  ∃ d : DigitMapping, equation_holds d

/-- 2003 is the smallest four-digit number SEEM for which there exists a valid mapping -/
theorem smallest_seem : (∃ d : DigitMapping, d.s = 2 ∧ d.e = 0 ∧ d.m = 3 ∧ equation_holds d) ∧
  (∀ n : Nat, n < 2003 → ¬∃ d : DigitMapping, 1000 * d.s + 100 * d.e + 10 * d.e + d.m = n ∧ equation_holds d) :=
sorry

end NUMINAMATH_CALUDE_smallest_seem_l2131_213178


namespace NUMINAMATH_CALUDE_earth_moon_distance_in_scientific_notation_l2131_213145

/-- The distance from Earth to the Moon's surface in kilometers -/
def earth_moon_distance : ℝ := 383900

/-- The scientific notation representation of the Earth-Moon distance -/
def earth_moon_distance_scientific : ℝ := 3.839 * (10 ^ 5)

theorem earth_moon_distance_in_scientific_notation :
  earth_moon_distance = earth_moon_distance_scientific :=
sorry

end NUMINAMATH_CALUDE_earth_moon_distance_in_scientific_notation_l2131_213145


namespace NUMINAMATH_CALUDE_tangent_line_intersects_ellipse_perpendicularly_l2131_213181

/-- An ellipse with semi-major axis 2 and semi-minor axis 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}

/-- A circle with radius 2√5/5 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = (2 * Real.sqrt 5 / 5)^2}

/-- A line tangent to the circle at point (m, n) -/
def TangentLine (m n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m * p.1 + n * p.2 = (2 * Real.sqrt 5 / 5)^2}

/-- The origin of the coordinate system -/
def Origin : ℝ × ℝ := (0, 0)

/-- Two points are perpendicular with respect to the origin -/
def Perpendicular (p q : ℝ × ℝ) : Prop :=
  p.1 * q.1 + p.2 * q.2 = 0

theorem tangent_line_intersects_ellipse_perpendicularly 
  (m n : ℝ) (h : (m, n) ∈ Circle) :
  ∃ (A B : ℝ × ℝ), A ∈ Ellipse ∧ B ∈ Ellipse ∧ 
    A ∈ TangentLine m n ∧ B ∈ TangentLine m n ∧
    Perpendicular (A.1 - Origin.1, A.2 - Origin.2) (B.1 - Origin.1, B.2 - Origin.2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersects_ellipse_perpendicularly_l2131_213181


namespace NUMINAMATH_CALUDE_math_club_challenge_l2131_213105

theorem math_club_challenge : ((7^2 - 5^2) - 2)^3 = 10648 := by
  sorry

end NUMINAMATH_CALUDE_math_club_challenge_l2131_213105


namespace NUMINAMATH_CALUDE_divides_fk_iff_divides_f_l2131_213134

theorem divides_fk_iff_divides_f (k : ℕ) (f : ℕ → ℕ) (x : ℕ) :
  (∀ n : ℕ, ∃ m : ℕ, f n = m * n) →
  (x ∣ f^[k] x ↔ x ∣ f x) :=
sorry

end NUMINAMATH_CALUDE_divides_fk_iff_divides_f_l2131_213134


namespace NUMINAMATH_CALUDE_triangle_inequality_from_inequality_l2131_213183

theorem triangle_inequality_from_inequality (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) :
  c < a + b ∧ a < b + c ∧ b < c + a := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_inequality_l2131_213183


namespace NUMINAMATH_CALUDE_unique_solution_l2131_213130

theorem unique_solution : 
  ∃! (x y z t : ℕ), 31 * (x * y * z * t + x * y + x * t + z * t + 1) = 40 * (y * z * t + y + t) ∧
  x = 1 ∧ y = 3 ∧ z = 2 ∧ t = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2131_213130


namespace NUMINAMATH_CALUDE_infinite_good_primes_infinite_non_good_primes_l2131_213131

/-- Definition of a good prime -/
def is_good_prime (p : ℕ) : Prop :=
  Prime p ∧ ∀ a b : ℕ, a > 0 → b > 0 → (a ≡ b [ZMOD p] ↔ a^3 ≡ b^3 [ZMOD p])

/-- The set of good primes is infinite -/
theorem infinite_good_primes : Set.Infinite {p : ℕ | is_good_prime p} :=
sorry

/-- The set of non-good primes is infinite -/
theorem infinite_non_good_primes : Set.Infinite {p : ℕ | Prime p ∧ ¬is_good_prime p} :=
sorry

end NUMINAMATH_CALUDE_infinite_good_primes_infinite_non_good_primes_l2131_213131


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2131_213111

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 + 6*x + 8 + (x + 4)*(x + 6) - 10
  ∃ x₁ x₂ : ℝ, x₁ = -4 + Real.sqrt 5 ∧ x₂ = -4 - Real.sqrt 5 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2131_213111


namespace NUMINAMATH_CALUDE_turtle_time_to_watering_hole_l2131_213187

/-- Represents the scenario of two lion cubs and a turtle moving towards a watering hole --/
structure WateringHoleScenario where
  /-- Speed of the first lion cub (in distance units per minute) --/
  speed_lion1 : ℝ
  /-- Distance of the first lion cub from the watering hole (in minutes) --/
  distance_lion1 : ℝ
  /-- Speed multiplier of the second lion cub relative to the first --/
  speed_multiplier_lion2 : ℝ
  /-- Distance of the turtle from the watering hole (in minutes) --/
  distance_turtle : ℝ

/-- Theorem stating the time it takes for the turtle to reach the watering hole after meeting the lion cubs --/
theorem turtle_time_to_watering_hole (scenario : WateringHoleScenario)
  (h1 : scenario.distance_lion1 = 5)
  (h2 : scenario.speed_multiplier_lion2 = 1.5)
  (h3 : scenario.distance_turtle = 30)
  (h4 : scenario.speed_lion1 > 0) :
  let meeting_time := 2
  let turtle_speed := 1 / scenario.distance_turtle
  let remaining_distance := 1 - meeting_time * turtle_speed
  remaining_distance * scenario.distance_turtle = 28 := by
  sorry

end NUMINAMATH_CALUDE_turtle_time_to_watering_hole_l2131_213187


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2131_213185

theorem fraction_equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (1 / (x - 2) = 3 / x) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2131_213185


namespace NUMINAMATH_CALUDE_largest_divisor_of_sequence_l2131_213106

theorem largest_divisor_of_sequence (n : ℕ) : ∃ (k : ℕ), k = 30 ∧ k ∣ (n^5 - n) ∧ ∀ m : ℕ, m > k → ¬(∀ n : ℕ, m ∣ (n^5 - n)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_sequence_l2131_213106


namespace NUMINAMATH_CALUDE_two_elements_condition_at_most_one_element_condition_l2131_213144

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x - 4 = 0}

-- Theorem for the first part of the problem
theorem two_elements_condition (a : ℝ) :
  (∃ x y : ℝ, x ∈ A a ∧ y ∈ A a ∧ x ≠ y) ↔ (a > -9/16 ∧ a ≠ 0) :=
sorry

-- Theorem for the second part of the problem
theorem at_most_one_element_condition (a : ℝ) :
  (∀ x y : ℝ, x ∈ A a → y ∈ A a → x = y) ↔ (a ≤ -9/16 ∨ a = 0) :=
sorry

end NUMINAMATH_CALUDE_two_elements_condition_at_most_one_element_condition_l2131_213144


namespace NUMINAMATH_CALUDE_smallest_multiple_l2131_213168

theorem smallest_multiple (n : ℕ) : n = 1767 ↔ 
  n > 0 ∧ 
  31 ∣ n ∧ 
  n % 97 = 3 ∧ 
  ∀ m : ℕ, m > 0 → 31 ∣ m → m % 97 = 3 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2131_213168


namespace NUMINAMATH_CALUDE_cosine_function_properties_l2131_213189

/-- Given a cosine function with specific properties, prove the value of ω and cos(α+β) -/
theorem cosine_function_properties (f : ℝ → ℝ) (ω α β : ℝ) :
  (∀ x, f x = 2 * Real.cos (ω * x + π / 6)) →
  ω > 0 →
  (∀ x, f (x + 10 * π) = f x) →
  (∀ y, y > 0 → y < 10 * π → ∀ x, f (x + y) ≠ f x) →
  α ∈ Set.Icc 0 (π / 2) →
  β ∈ Set.Icc 0 (π / 2) →
  f (5 * α + 5 * π / 3) = -6 / 5 →
  f (5 * β - 5 * π / 6) = 16 / 17 →
  ω = 1 / 5 ∧ Real.cos (α + β) = -13 / 85 := by
  sorry


end NUMINAMATH_CALUDE_cosine_function_properties_l2131_213189


namespace NUMINAMATH_CALUDE_sum_of_products_l2131_213171

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 941) 
  (h2 : a + b + c = 31) : 
  a*b + b*c + c*a = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2131_213171


namespace NUMINAMATH_CALUDE_juans_number_problem_l2131_213182

theorem juans_number_problem (n : ℝ) : 
  (((n + 3) * 2 - 2) / 2 = 8) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_juans_number_problem_l2131_213182


namespace NUMINAMATH_CALUDE_sum_of_n_values_l2131_213143

theorem sum_of_n_values (m n : ℕ+) : 
  (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 5 →
  ∃ (n₁ n₂ n₃ : ℕ+), 
    (∀ k : ℕ+, ((1 : ℚ) / m + (1 : ℚ) / k = (1 : ℚ) / 5) → (k = n₁ ∨ k = n₂ ∨ k = n₃)) ∧
    n₁.val + n₂.val + n₃.val = 46 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_n_values_l2131_213143


namespace NUMINAMATH_CALUDE_intersection_point_solution_and_b_value_l2131_213142

/-- The intersection point of two lines -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- A linear function of the form y = mx + c -/
structure LinearFunction where
  m : ℝ
  c : ℝ

/-- Given two linear functions and their intersection point, prove the solution and b value -/
theorem intersection_point_solution_and_b_value 
  (f1 : LinearFunction)
  (f2 : LinearFunction)
  (P : IntersectionPoint)
  (h1 : f1.m = 2 ∧ f1.c = -5)
  (h2 : f2.m = 3)
  (h3 : P.x = 1 ∧ P.y = -3)
  (h4 : P.y = f1.m * P.x + f1.c)
  (h5 : P.y = f2.m * P.x + f2.c) :
  (∃ (x y : ℝ), x = 1 ∧ y = -3 ∧ 
    y = f1.m * x + f1.c ∧
    y = f2.m * x + f2.c) ∧
  f2.c = -6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_solution_and_b_value_l2131_213142


namespace NUMINAMATH_CALUDE_river_trip_longer_than_lake_trip_l2131_213151

theorem river_trip_longer_than_lake_trip 
  (a b S : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : b < a) 
  (hS : S > 0) : 
  (2 * a * S) / (a^2 - b^2) > (2 * S) / a := by
  sorry

end NUMINAMATH_CALUDE_river_trip_longer_than_lake_trip_l2131_213151


namespace NUMINAMATH_CALUDE_alcohol_volume_bound_l2131_213148

/-- Represents the volume of pure alcohol in container B after n operations -/
def alcohol_volume (x y z : ℝ) (n : ℕ+) : ℝ :=
  sorry

/-- Theorem stating that the volume of pure alcohol in container B 
    is always less than or equal to xy/(x+y) -/
theorem alcohol_volume_bound (x y z : ℝ) (n : ℕ+) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxy : x < z) (hyz : y < z) :
  alcohol_volume x y z n ≤ (x * y) / (x + y) :=
sorry

end NUMINAMATH_CALUDE_alcohol_volume_bound_l2131_213148


namespace NUMINAMATH_CALUDE_absolute_value_sqrt_problem_l2131_213196

theorem absolute_value_sqrt_problem : |-2 * Real.sqrt 2| - Real.sqrt 4 * Real.sqrt 2 + (π - 5)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sqrt_problem_l2131_213196


namespace NUMINAMATH_CALUDE_strategy_is_injective_l2131_213141

-- Define the set of possible numbers
inductive Number : Type
| one : Number
| two : Number
| three : Number

-- Define the set of possible answers
inductive Answer : Type
| yes : Answer
| no : Answer
| dontKnow : Answer

-- Define the strategy function
def strategy : Number → Answer
| Number.one => Answer.yes
| Number.two => Answer.dontKnow
| Number.three => Answer.no

-- Theorem: The strategy function is injective
theorem strategy_is_injective :
  ∀ x y : Number, x ≠ y → strategy x ≠ strategy y := by
  sorry

#check strategy_is_injective

end NUMINAMATH_CALUDE_strategy_is_injective_l2131_213141


namespace NUMINAMATH_CALUDE_coin_stack_arrangements_l2131_213108

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of possible face arrangements for n coins where no two adjacent coins are face to face -/
def faceArrangements (n : ℕ) : ℕ := n + 1

theorem coin_stack_arrangements :
  let totalCoins : ℕ := 8
  let goldCoins : ℕ := 5
  let silverCoins : ℕ := 3
  (binomial totalCoins goldCoins) * (faceArrangements totalCoins) = 504 := by sorry

end NUMINAMATH_CALUDE_coin_stack_arrangements_l2131_213108


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2131_213126

/-- Given that i^2 = -1, prove that (3-2i)/(4+5i) = 2/41 - (23/41)i -/
theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - 2*i) / (4 + 5*i) = 2/41 - (23/41)*i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2131_213126


namespace NUMINAMATH_CALUDE_a8_min_value_l2131_213124

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 3 + a 6 = a 4 + 5
  a2_bound : a 2 ≤ 1

/-- The minimum value of the 8th term in the arithmetic sequence is 9 -/
theorem a8_min_value (seq : ArithmeticSequence) : seq.a 8 ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_a8_min_value_l2131_213124


namespace NUMINAMATH_CALUDE_wings_count_l2131_213165

/-- Calculates the total number of wings for birds bought with money from grandparents -/
def total_wings (num_grandparents : ℕ) (money_per_grandparent : ℕ) (cost_per_bird : ℕ) (wings_per_bird : ℕ) : ℕ :=
  let total_money := num_grandparents * money_per_grandparent
  let num_birds := total_money / cost_per_bird
  num_birds * wings_per_bird

/-- Theorem: Given the problem conditions, the total number of wings is 20 -/
theorem wings_count :
  total_wings 4 50 20 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_wings_count_l2131_213165
