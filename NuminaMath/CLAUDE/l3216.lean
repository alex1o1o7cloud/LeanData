import Mathlib

namespace NUMINAMATH_CALUDE_logarithm_expression_equals_two_plus_pi_l3216_321601

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_two_plus_pi :
  lg 5 * (Real.log 20 / Real.log (Real.sqrt 10)) + (lg (2 ^ Real.sqrt 2))^2 + Real.exp (Real.log π) = 2 + π := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_two_plus_pi_l3216_321601


namespace NUMINAMATH_CALUDE_min_value_expression_l3216_321635

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b) * (b + 4/a) ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ (a₀ + 1/b₀) * (b₀ + 4/a₀) = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3216_321635


namespace NUMINAMATH_CALUDE_sqrt_a_squared_b_l3216_321669

theorem sqrt_a_squared_b (a b : ℝ) (h : a * b < 0) : Real.sqrt (a^2 * b) = -a * Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_squared_b_l3216_321669


namespace NUMINAMATH_CALUDE_power_product_equals_75600_l3216_321628

theorem power_product_equals_75600 : 2^4 * 5^2 * 3^3 * 7 = 75600 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_75600_l3216_321628


namespace NUMINAMATH_CALUDE_line_through_points_l3216_321647

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) -/
def line_equation (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

/-- Theorem: The equation of the line passing through (1, 0) and (0, 1) is x + y - 1 = 0 -/
theorem line_through_points : 
  ∀ x y : ℝ, line_equation 1 0 0 1 x y ↔ x + y - 1 = 0 := by
  sorry

#check line_through_points

end NUMINAMATH_CALUDE_line_through_points_l3216_321647


namespace NUMINAMATH_CALUDE_reflection_line_sum_l3216_321615

/-- Given a line y = mx + b, if the reflection of point (2, 3) across this line is (10, 7), then m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    (x - 2)^2 + (y - 3)^2 = (10 - x)^2 + (7 - y)^2 ∧ 
    y = m * x + b ∧
    (y - 3) = -1 / m * (x - 2)) → 
  m + b = 15 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l3216_321615


namespace NUMINAMATH_CALUDE_journey_time_l3216_321691

/-- Represents a journey with constant speed -/
structure Journey where
  quarter_time : ℝ  -- Time to cover 1/4 of the journey
  third_time : ℝ    -- Time to cover 1/3 of the journey

/-- The total time for the journey -/
def total_time (j : Journey) : ℝ :=
  (j.third_time - j.quarter_time) * 12 - j.quarter_time

/-- Theorem stating that for the given journey, the total time is 280 minutes -/
theorem journey_time (j : Journey) 
  (h1 : j.quarter_time = 20) 
  (h2 : j.third_time = 45) : 
  total_time j = 280 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_l3216_321691


namespace NUMINAMATH_CALUDE_definite_integral_x_squared_plus_sqrt_one_minus_x_squared_l3216_321652

theorem definite_integral_x_squared_plus_sqrt_one_minus_x_squared :
  ∫ x in (-1)..1, (x^2 + Real.sqrt (1 - x^2)) = 2/3 + π/2 := by sorry

end NUMINAMATH_CALUDE_definite_integral_x_squared_plus_sqrt_one_minus_x_squared_l3216_321652


namespace NUMINAMATH_CALUDE_remaining_questions_to_write_l3216_321641

theorem remaining_questions_to_write
  (total_multiple_choice : ℕ)
  (total_problem_solving : ℕ)
  (total_true_false : ℕ)
  (fraction_multiple_choice_written : ℚ)
  (fraction_problem_solving_written : ℚ)
  (fraction_true_false_written : ℚ)
  (h1 : total_multiple_choice = 35)
  (h2 : total_problem_solving = 15)
  (h3 : total_true_false = 20)
  (h4 : fraction_multiple_choice_written = 3/7)
  (h5 : fraction_problem_solving_written = 1/5)
  (h6 : fraction_true_false_written = 1/4) :
  (total_multiple_choice - (fraction_multiple_choice_written * total_multiple_choice).num) +
  (total_problem_solving - (fraction_problem_solving_written * total_problem_solving).num) +
  (total_true_false - (fraction_true_false_written * total_true_false).num) = 47 :=
by sorry

end NUMINAMATH_CALUDE_remaining_questions_to_write_l3216_321641


namespace NUMINAMATH_CALUDE_total_hot_dog_cost_l3216_321638

def hot_dog_cost (group : ℕ) (quantity : ℕ) (price : ℚ) : ℚ :=
  quantity * price

theorem total_hot_dog_cost : 
  let group1_cost := hot_dog_cost 1 4 0.60
  let group2_cost := hot_dog_cost 2 5 0.75
  let group3_cost := hot_dog_cost 3 3 0.90
  group1_cost + group2_cost + group3_cost = 8.85 := by
  sorry

end NUMINAMATH_CALUDE_total_hot_dog_cost_l3216_321638


namespace NUMINAMATH_CALUDE_intersection_M_N_l3216_321660

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 - x - 2 ≤ 0}

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 ∪ {2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3216_321660


namespace NUMINAMATH_CALUDE_negation_of_implication_l3216_321626

theorem negation_of_implication (a b : ℝ) :
  ¬(a = 0 ∧ b = 0 → a^2 + b^2 = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3216_321626


namespace NUMINAMATH_CALUDE_three_digit_reverse_subtraction_l3216_321604

theorem three_digit_reverse_subtraction (b c : ℕ) : 
  (0 < c) ∧ (c < 10) ∧ (b < 10) → 
  (101*c + 10*b + 300) - (101*c + 10*b + 3) = 297 := by
  sorry

#check three_digit_reverse_subtraction

end NUMINAMATH_CALUDE_three_digit_reverse_subtraction_l3216_321604


namespace NUMINAMATH_CALUDE_min_points_dodecahedron_correct_min_points_icosahedron_correct_l3216_321640

/-- A dodecahedron is a polyhedron with 12 faces, where each face is a regular pentagon and each vertex belongs to 3 faces. -/
structure Dodecahedron where
  faces : ℕ
  faces_are_pentagons : Bool
  vertex_face_count : ℕ
  h_faces : faces = 12
  h_pentagons : faces_are_pentagons = true
  h_vertex : vertex_face_count = 3

/-- An icosahedron is a polyhedron with 20 faces and 12 vertices, where each face is an equilateral triangle. -/
structure Icosahedron where
  faces : ℕ
  vertices : ℕ
  faces_are_triangles : Bool
  h_faces : faces = 20
  h_vertices : vertices = 12
  h_triangles : faces_are_triangles = true

/-- The minimum number of points that must be marked on the surface of a dodecahedron
    so that there is at least one marked point on each face. -/
def min_points_dodecahedron (d : Dodecahedron) : ℕ := 4

/-- The minimum number of points that must be marked on the surface of an icosahedron
    so that there is at least one marked point on each face. -/
def min_points_icosahedron (i : Icosahedron) : ℕ := 6

/-- Theorem stating the minimum number of points for a dodecahedron. -/
theorem min_points_dodecahedron_correct (d : Dodecahedron) :
  min_points_dodecahedron d = 4 := by sorry

/-- Theorem stating the minimum number of points for an icosahedron. -/
theorem min_points_icosahedron_correct (i : Icosahedron) :
  min_points_icosahedron i = 6 := by sorry

end NUMINAMATH_CALUDE_min_points_dodecahedron_correct_min_points_icosahedron_correct_l3216_321640


namespace NUMINAMATH_CALUDE_library_books_count_l3216_321651

theorem library_books_count (num_bookshelves : ℕ) (floors_per_bookshelf : ℕ) (books_per_floor : ℕ) :
  num_bookshelves = 28 →
  floors_per_bookshelf = 6 →
  books_per_floor = 19 →
  num_bookshelves * floors_per_bookshelf * books_per_floor = 3192 :=
by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l3216_321651


namespace NUMINAMATH_CALUDE_inequality_preservation_l3216_321687

theorem inequality_preservation (a b c : ℝ) (h : a < b) : a - c < b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3216_321687


namespace NUMINAMATH_CALUDE_roots_sum_squares_l3216_321699

theorem roots_sum_squares (r s : ℝ) : 
  r^2 - 5*r + 6 = 0 → s^2 - 5*s + 6 = 0 → r^2 + s^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_squares_l3216_321699


namespace NUMINAMATH_CALUDE_computer_printer_price_l3216_321689

/-- The total price of a basic computer and printer, given specific conditions -/
theorem computer_printer_price (basic_price enhanced_price printer_price total_price : ℝ) : 
  basic_price = 2125 →
  enhanced_price = basic_price + 500 →
  printer_price = (1 / 8) * (enhanced_price + printer_price) →
  total_price = basic_price + printer_price →
  total_price = 2500 := by
  sorry

end NUMINAMATH_CALUDE_computer_printer_price_l3216_321689


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l3216_321629

theorem fraction_equals_zero (x : ℝ) (h : 6 * x ≠ 0) :
  (x - 5) / (6 * x) = 0 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l3216_321629


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3216_321697

theorem exponent_multiplication (x : ℝ) (m n : ℕ) :
  x^m * x^n = x^(m + n) := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3216_321697


namespace NUMINAMATH_CALUDE_freshmen_psych_liberal_arts_percentage_is_four_l3216_321664

/-- The percentage of students who are freshmen psychology majors in the School of Liberal Arts -/
def freshmen_psych_liberal_arts_percentage (total_students : ℕ) : ℚ :=
  let freshmen_percentage : ℚ := 50 / 100
  let liberal_arts_percentage : ℚ := 40 / 100
  let psychology_percentage : ℚ := 20 / 100
  freshmen_percentage * liberal_arts_percentage * psychology_percentage * 100

theorem freshmen_psych_liberal_arts_percentage_is_four (total_students : ℕ) :
  freshmen_psych_liberal_arts_percentage total_students = 4 := by
  sorry

end NUMINAMATH_CALUDE_freshmen_psych_liberal_arts_percentage_is_four_l3216_321664


namespace NUMINAMATH_CALUDE_product_of_1011_2_and_102_3_l3216_321658

def base_2_to_10 (n : ℕ) : ℕ := sorry

def base_3_to_10 (n : ℕ) : ℕ := sorry

theorem product_of_1011_2_and_102_3 : 
  (base_2_to_10 1011) * (base_3_to_10 102) = 121 := by sorry

end NUMINAMATH_CALUDE_product_of_1011_2_and_102_3_l3216_321658


namespace NUMINAMATH_CALUDE_students_not_enrolled_l3216_321627

theorem students_not_enrolled (total : ℕ) (english : ℕ) (history : ℕ) (both : ℕ) : 
  total = 60 → english = 42 → history = 30 → both = 18 →
  total - (english + history - both) = 6 := by
sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l3216_321627


namespace NUMINAMATH_CALUDE_tan_fraction_equality_l3216_321613

theorem tan_fraction_equality (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_fraction_equality_l3216_321613


namespace NUMINAMATH_CALUDE_rachels_to_christines_ratio_l3216_321677

def strawberries_per_pie : ℕ := 3
def christines_strawberries : ℕ := 10
def total_pies : ℕ := 10

theorem rachels_to_christines_ratio :
  let total_strawberries := strawberries_per_pie * total_pies
  let rachels_strawberries := total_strawberries - christines_strawberries
  (rachels_strawberries : ℚ) / christines_strawberries = 2 := by sorry

end NUMINAMATH_CALUDE_rachels_to_christines_ratio_l3216_321677


namespace NUMINAMATH_CALUDE_square_pentagon_exterior_angle_l3216_321684

/-- The exterior angle formed by a square and a regular pentagon sharing a common side --/
def exteriorAngle (n : ℕ) : ℚ :=
  360 - (180 * (n - 2) / n) - 90

/-- Theorem: The exterior angle BAC formed by a square and a regular pentagon sharing a common side AD is 162° --/
theorem square_pentagon_exterior_angle :
  exteriorAngle 5 = 162 := by
  sorry

end NUMINAMATH_CALUDE_square_pentagon_exterior_angle_l3216_321684


namespace NUMINAMATH_CALUDE_statement_equivalence_l3216_321663

theorem statement_equivalence (P Q : Prop) : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by
  sorry

end NUMINAMATH_CALUDE_statement_equivalence_l3216_321663


namespace NUMINAMATH_CALUDE_number_plus_thrice_value_l3216_321623

theorem number_plus_thrice_value (x : ℕ) (value : ℕ) : x = 5 → x + 3 * x = value → value = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_thrice_value_l3216_321623


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3216_321616

theorem complex_modulus_problem (a b : ℝ) :
  (a + Complex.I) * (1 - Complex.I) = 3 + b * Complex.I →
  Complex.abs (a + b * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3216_321616


namespace NUMINAMATH_CALUDE_flu_infection_spread_l3216_321605

/-- The average number of people infected by one person in each round of infection -/
def average_infections : ℕ := 13

/-- The number of rounds of infection -/
def num_rounds : ℕ := 2

/-- The total number of people infected after two rounds -/
def total_infected : ℕ := 196

/-- The number of initially infected people -/
def initial_infected : ℕ := 1

theorem flu_infection_spread :
  (initial_infected + average_infections * initial_infected + 
   average_infections * (initial_infected + average_infections * initial_infected) = total_infected) ∧
  (average_infections > 0) := by
  sorry

end NUMINAMATH_CALUDE_flu_infection_spread_l3216_321605


namespace NUMINAMATH_CALUDE_odd_monotonous_unique_zero_implies_k_is_quarter_l3216_321633

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is monotonous if it's either increasing or decreasing -/
def IsMonotonous (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∨ (∀ x y, x < y → f x > f y)

/-- A function has only one zero point if there exists exactly one x such that f(x) = 0 -/
def HasUniqueZero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

theorem odd_monotonous_unique_zero_implies_k_is_quarter
    (f : ℝ → ℝ) (k : ℝ)
    (h_odd : IsOdd f)
    (h_monotonous : IsMonotonous f)
    (h_unique_zero : HasUniqueZero (fun x ↦ f (x^2) + f (k - x))) :
    k = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_odd_monotonous_unique_zero_implies_k_is_quarter_l3216_321633


namespace NUMINAMATH_CALUDE_days_to_read_book_290_l3216_321611

/-- Calculates the number of days required to read a book with the given reading pattern -/
def daysToReadBook (totalPages : ℕ) (sundayPages : ℕ) (otherDayPages : ℕ) : ℕ :=
  let pagesPerWeek := sundayPages + 6 * otherDayPages
  let completeWeeks := totalPages / pagesPerWeek
  let remainingPages := totalPages % pagesPerWeek
  let additionalDays := 
    if remainingPages ≤ sundayPages 
    then 1
    else 1 + ((remainingPages - sundayPages) + (otherDayPages - 1)) / otherDayPages
  7 * completeWeeks + additionalDays

/-- Theorem stating that it takes 41 days to read a 290-page book with the given reading pattern -/
theorem days_to_read_book_290 : 
  daysToReadBook 290 25 4 = 41 := by
  sorry

end NUMINAMATH_CALUDE_days_to_read_book_290_l3216_321611


namespace NUMINAMATH_CALUDE_difference_x_y_l3216_321621

theorem difference_x_y (x y : ℤ) (h1 : x + y = 20) (h2 : x = 28) : x - y = 36 := by
  sorry

end NUMINAMATH_CALUDE_difference_x_y_l3216_321621


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_specific_numbers_l3216_321662

theorem arithmetic_mean_of_specific_numbers :
  let numbers : List ℝ := [-5, 3.5, 12, 20]
  (numbers.sum / numbers.length : ℝ) = 7.625 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_specific_numbers_l3216_321662


namespace NUMINAMATH_CALUDE_rachel_money_left_l3216_321624

theorem rachel_money_left (earnings : ℝ) (lunch_fraction : ℝ) (dvd_fraction : ℝ) : 
  earnings = 200 →
  lunch_fraction = 1/4 →
  dvd_fraction = 1/2 →
  earnings - (lunch_fraction * earnings + dvd_fraction * earnings) = 50 := by
sorry

end NUMINAMATH_CALUDE_rachel_money_left_l3216_321624


namespace NUMINAMATH_CALUDE_sum_digits_888_base_8_l3216_321625

/-- Represents a number in base 8 as a list of digits (least significant digit first) -/
def BaseEightRepresentation := List Nat

/-- Converts a natural number from base 10 to base 8 -/
def toBaseEight (n : Nat) : BaseEightRepresentation :=
  sorry

/-- Calculates the sum of digits in a base 8 representation -/
def sumDigits (repr : BaseEightRepresentation) : Nat :=
  sorry

theorem sum_digits_888_base_8 :
  sumDigits (toBaseEight 888) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_888_base_8_l3216_321625


namespace NUMINAMATH_CALUDE_tan_beta_value_l3216_321661

theorem tan_beta_value (α β : Real) 
  (h1 : (Real.sin α * Real.cos α) / (Real.cos (2 * α) + 1) = 1)
  (h2 : Real.tan (α - β) = 3) : 
  Real.tan β = -1/7 := by sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3216_321661


namespace NUMINAMATH_CALUDE_fathers_age_ratio_l3216_321679

theorem fathers_age_ratio (R : ℕ) : 
  let F := 4 * R
  let father_age_after_8 := F + 8
  let ronit_age_after_8 := R + 8
  let father_age_after_16 := F + 16
  let ronit_age_after_16 := R + 16
  (∃ M : ℕ, father_age_after_8 = M * ronit_age_after_8) ∧ 
  (father_age_after_16 = 2 * ronit_age_after_16) →
  (father_age_after_8 : ℚ) / ronit_age_after_8 = 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_fathers_age_ratio_l3216_321679


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3216_321653

def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 6 > 0}

theorem intersection_of_M_and_N :
  M ∩ N = {x | -4 ≤ x ∧ x < -2} ∪ {x | 3 < x ∧ x ≤ 7} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3216_321653


namespace NUMINAMATH_CALUDE_square_position_after_2023_transformations_l3216_321618

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | CDAB
  | DCBA
  | BADC

-- Define the transformations
def rotate180 (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.CDAB
  | SquarePosition.CDAB => SquarePosition.ABCD
  | SquarePosition.DCBA => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.DCBA

def reflectHorizontal (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.DCBA
  | SquarePosition.CDAB => SquarePosition.BADC
  | SquarePosition.DCBA => SquarePosition.ABCD
  | SquarePosition.BADC => SquarePosition.CDAB

-- Define the alternating transformations
def alternateTransform (n : Nat) (pos : SquarePosition) : SquarePosition :=
  match n with
  | 0 => pos
  | n + 1 => 
    if n % 2 == 0
    then reflectHorizontal (alternateTransform n pos)
    else rotate180 (alternateTransform n pos)

-- The theorem to prove
theorem square_position_after_2023_transformations :
  alternateTransform 2023 SquarePosition.ABCD = SquarePosition.DCBA := by
  sorry


end NUMINAMATH_CALUDE_square_position_after_2023_transformations_l3216_321618


namespace NUMINAMATH_CALUDE_det_is_zero_l3216_321600

variables {α : Type*} [Field α]
variables (s p q : α)

-- Define the polynomial
def f (x : α) := x^3 - s*x^2 + p*x + q

-- Define the roots
structure Roots (s p q : α) where
  a : α
  b : α
  c : α
  root_a : f s p q a = 0
  root_b : f s p q b = 0
  root_c : f s p q c = 0

-- Define the matrix
def matrix (r : Roots s p q) : Matrix (Fin 3) (Fin 3) α :=
  ![![r.a, r.b, r.c],
    ![r.c, r.a, r.b],
    ![r.b, r.c, r.a]]

-- Theorem statement
theorem det_is_zero (r : Roots s p q) : 
  Matrix.det (matrix s p q r) = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_is_zero_l3216_321600


namespace NUMINAMATH_CALUDE_rebus_solution_l3216_321608

theorem rebus_solution : ∃! (A B C : ℕ), 
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) ∧ 
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧
  A = 4 ∧ B = 7 ∧ C = 6 := by
sorry

end NUMINAMATH_CALUDE_rebus_solution_l3216_321608


namespace NUMINAMATH_CALUDE_complex_quotient_real_l3216_321607

theorem complex_quotient_real (t : ℝ) : 
  let z₁ : ℂ := 2*t + Complex.I
  let z₂ : ℂ := 1 - 2*Complex.I
  (∃ (r : ℝ), z₁ / z₂ = r) → t = -1/4 := by
sorry

end NUMINAMATH_CALUDE_complex_quotient_real_l3216_321607


namespace NUMINAMATH_CALUDE_expression_is_factored_l3216_321649

/-- Represents a quadratic expression of the form ax^2 + bx + c -/
structure QuadraticExpression (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- Represents a factored quadratic expression of the form (x - r)^2 -/
structure FactoredQuadratic (α : Type*) [Ring α] where
  r : α

/-- Checks if a quadratic expression is factored from left to right -/
def is_factored_left_to_right {α : Type*} [Ring α] (q : QuadraticExpression α) (f : FactoredQuadratic α) : Prop :=
  q.a = 1 ∧ q.b = -2 * f.r ∧ q.c = f.r^2

/-- The given quadratic expression x^2 - 6x + 9 -/
def given_expression : QuadraticExpression ℤ := ⟨1, -6, 9⟩

/-- The factored form (x - 3)^2 -/
def factored_form : FactoredQuadratic ℤ := ⟨3⟩

/-- Theorem stating that the given expression represents factorization from left to right -/
theorem expression_is_factored : is_factored_left_to_right given_expression factored_form := by
  sorry

end NUMINAMATH_CALUDE_expression_is_factored_l3216_321649


namespace NUMINAMATH_CALUDE_sale_recording_l3216_321632

/-- Represents the inventory change for a given number of items. -/
def inventoryChange (items : ℤ) : ℤ := items

/-- The bookkeeping convention for recording purchases. -/
axiom purchase_convention (items : ℕ) : inventoryChange items = items

/-- Theorem: The sale of 5 items should be recorded as -5. -/
theorem sale_recording : inventoryChange (-5) = -5 := by
  sorry

end NUMINAMATH_CALUDE_sale_recording_l3216_321632


namespace NUMINAMATH_CALUDE_product_of_solutions_l3216_321619

theorem product_of_solutions (x : ℝ) : 
  (∃ α β : ℝ, x^2 + 4*x + 49 = 0 ∧ x = α ∨ x = β) → 
  (∃ α β : ℝ, x^2 + 4*x + 49 = 0 ∧ x = α ∨ x = β ∧ α * β = -49) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3216_321619


namespace NUMINAMATH_CALUDE_same_name_pair_exists_l3216_321645

theorem same_name_pair_exists (n : ℕ) (h_n : n = 33) :
  ∀ (first_name_groups last_name_groups : Fin n → Fin 11),
    (∀ i : Fin 11, ∃ j : Fin n, first_name_groups j = i) →
    (∀ i : Fin 11, ∃ j : Fin n, last_name_groups j = i) →
    ∃ x y : Fin n, x ≠ y ∧ first_name_groups x = first_name_groups y ∧ last_name_groups x = last_name_groups y :=
by
  sorry

#check same_name_pair_exists

end NUMINAMATH_CALUDE_same_name_pair_exists_l3216_321645


namespace NUMINAMATH_CALUDE_hotel_bubble_bath_l3216_321610

/-- The amount of bubble bath needed for a hotel with couples and single rooms -/
def bubble_bath_needed (couple_rooms single_rooms : ℕ) (bath_per_person : ℕ) : ℕ :=
  (2 * couple_rooms + single_rooms) * bath_per_person

/-- Theorem: The amount of bubble bath needed for 13 couple rooms and 14 single rooms is 400ml -/
theorem hotel_bubble_bath :
  bubble_bath_needed 13 14 10 = 400 := by
  sorry

end NUMINAMATH_CALUDE_hotel_bubble_bath_l3216_321610


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3216_321686

theorem simplify_and_rationalize :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (Real.sqrt 8 / Real.sqrt 10) * (Real.sqrt 5 / Real.sqrt 12) / (Real.sqrt 9 / Real.sqrt 14) = 
  Real.sqrt a / b ∧
  a = 28 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3216_321686


namespace NUMINAMATH_CALUDE_line_circle_intersection_slope_range_l3216_321602

/-- Given a line passing through (4,0) and intersecting the circle (x-2)^2 + y^2 = 1,
    prove that its slope k is between -√3/3 and √3/3 inclusive. -/
theorem line_circle_intersection_slope_range :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), y = k * (x - 4) ∧ (x - 2)^2 + y^2 = 1) →
  -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_slope_range_l3216_321602


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l3216_321676

/-- A triangle with given inscribed circle radius, circumscribed circle radius, and one angle. -/
structure Triangle where
  r : ℝ  -- radius of inscribed circle
  R : ℝ  -- radius of circumscribed circle
  α : ℝ  -- one angle of the triangle (in radians)

/-- Theorem stating the conditions for the existence of a triangle with given parameters. -/
theorem triangle_existence_condition (t : Triangle) :
  (∃ (triangle : Triangle), triangle = t) ↔ 
  (0 < t.α ∧ t.α < Real.pi ∧ t.R ≥ 2 * t.r) :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l3216_321676


namespace NUMINAMATH_CALUDE_fish_offspring_conversion_l3216_321630

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- The fish offspring count in base 7 --/
def fishOffspringBase7 : ℕ := 265

theorem fish_offspring_conversion :
  base7ToBase10 fishOffspringBase7 = 145 := by
  sorry

end NUMINAMATH_CALUDE_fish_offspring_conversion_l3216_321630


namespace NUMINAMATH_CALUDE_ribbon_count_l3216_321646

theorem ribbon_count (morning_given afternoon_given remaining : ℕ) 
  (h1 : morning_given = 14)
  (h2 : afternoon_given = 16)
  (h3 : remaining = 8) :
  morning_given + afternoon_given + remaining = 38 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_count_l3216_321646


namespace NUMINAMATH_CALUDE_flowers_given_to_brother_correct_flowers_given_l3216_321671

theorem flowers_given_to_brother (amanda_flowers : ℕ) (peter_flowers_left : ℕ) : ℕ :=
  let peter_initial_flowers := 3 * amanda_flowers
  peter_initial_flowers - peter_flowers_left

theorem correct_flowers_given (amanda_flowers : ℕ) (peter_flowers_left : ℕ)
    (h1 : amanda_flowers = 20)
    (h2 : peter_flowers_left = 45) :
    flowers_given_to_brother amanda_flowers peter_flowers_left = 15 := by
  sorry

end NUMINAMATH_CALUDE_flowers_given_to_brother_correct_flowers_given_l3216_321671


namespace NUMINAMATH_CALUDE_first_half_speed_l3216_321642

/-- Given a journey with the following properties:
  * The total distance is 224 km
  * The total time is 10 hours
  * The second half of the journey is traveled at 24 km/hr
  Prove that the speed during the first half of the journey is 21 km/hr -/
theorem first_half_speed
  (total_distance : ℝ)
  (total_time : ℝ)
  (second_half_speed : ℝ)
  (h1 : total_distance = 224)
  (h2 : total_time = 10)
  (h3 : second_half_speed = 24)
  : (total_distance / 2) / (total_time - (total_distance / 2) / second_half_speed) = 21 := by
  sorry

end NUMINAMATH_CALUDE_first_half_speed_l3216_321642


namespace NUMINAMATH_CALUDE_balloon_height_calculation_l3216_321667

theorem balloon_height_calculation (initial_budget : ℚ) (sheet_cost : ℚ) (rope_cost : ℚ) (propane_cost : ℚ) (helium_price_per_oz : ℚ) (height_per_oz : ℚ) : 
  initial_budget = 200 →
  sheet_cost = 42 →
  rope_cost = 18 →
  propane_cost = 14 →
  helium_price_per_oz = 3/2 →
  height_per_oz = 113 →
  ((initial_budget - sheet_cost - rope_cost - propane_cost) / helium_price_per_oz) * height_per_oz = 9492 :=
by sorry

end NUMINAMATH_CALUDE_balloon_height_calculation_l3216_321667


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3216_321682

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 5*x + 3 ≤ 0) ↔ (∃ x : ℝ, x^2 - 5*x + 3 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3216_321682


namespace NUMINAMATH_CALUDE_rainfall_problem_l3216_321620

/-- Rainfall problem -/
theorem rainfall_problem (sunday monday tuesday : ℝ) 
  (h1 : tuesday = 2 * monday)
  (h2 : monday = sunday + 3)
  (h3 : sunday + monday + tuesday = 25) :
  sunday = 4 := by
sorry

end NUMINAMATH_CALUDE_rainfall_problem_l3216_321620


namespace NUMINAMATH_CALUDE_count_even_factors_l3216_321681

def n : ℕ := 2^4 * 3^3 * 5^2

/-- The number of even positive factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem count_even_factors :
  num_even_factors n = 48 := by sorry

end NUMINAMATH_CALUDE_count_even_factors_l3216_321681


namespace NUMINAMATH_CALUDE_constant_dot_product_l3216_321692

-- Define the curve E
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define point D
def D : ℝ × ℝ := (-2, 0)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the vector from D to a point
def vector_DA (A : ℝ × ℝ) : ℝ × ℝ := (A.1 - D.1, A.2 - D.2)

theorem constant_dot_product :
  ∀ A B : ℝ × ℝ, A ∈ E → B ∈ E →
  dot_product (vector_DA A) (vector_DA B) = 3 := by sorry

end NUMINAMATH_CALUDE_constant_dot_product_l3216_321692


namespace NUMINAMATH_CALUDE_granger_age_is_42_l3216_321672

/-- Mr. Granger's current age -/
def granger_age : ℕ := sorry

/-- Mr. Granger's son's current age -/
def son_age : ℕ := sorry

/-- First condition: Mr. Granger's age is 10 years more than twice his son's age -/
axiom condition1 : granger_age = 2 * son_age + 10

/-- Second condition: Last year, Mr. Granger's age was 4 years less than 3 times his son's age -/
axiom condition2 : granger_age - 1 = 3 * (son_age - 1) - 4

/-- Theorem: Mr. Granger's age is 42 years -/
theorem granger_age_is_42 : granger_age = 42 := by sorry

end NUMINAMATH_CALUDE_granger_age_is_42_l3216_321672


namespace NUMINAMATH_CALUDE_reservoir_water_supply_l3216_321637

/-- Reservoir water supply problem -/
theorem reservoir_water_supply
  (reservoir_volume : ℝ)
  (initial_population : ℝ)
  (initial_sustainability : ℝ)
  (new_population : ℝ)
  (new_sustainability : ℝ)
  (h_reservoir : reservoir_volume = 120)
  (h_initial_pop : initial_population = 160000)
  (h_initial_sus : initial_sustainability = 20)
  (h_new_pop : new_population = 200000)
  (h_new_sus : new_sustainability = 15) :
  ∃ (annual_precipitation : ℝ) (annual_consumption_pp : ℝ),
    annual_precipitation = 200 ∧
    annual_consumption_pp = 50 ∧
    reservoir_volume + initial_sustainability * annual_precipitation = initial_population * initial_sustainability * annual_consumption_pp / 1000000 ∧
    reservoir_volume + new_sustainability * annual_precipitation = new_population * new_sustainability * annual_consumption_pp / 1000000 :=
by sorry


end NUMINAMATH_CALUDE_reservoir_water_supply_l3216_321637


namespace NUMINAMATH_CALUDE_death_rate_calculation_l3216_321659

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate (people per two seconds) -/
def birth_rate : ℕ := 7

/-- Represents the net population increase per day -/
def net_increase_per_day : ℕ := 259200

/-- Represents the death rate (people per two seconds) -/
def death_rate : ℕ := 1

theorem death_rate_calculation :
  (birth_rate - death_rate) * seconds_per_day / 2 = net_increase_per_day :=
sorry

end NUMINAMATH_CALUDE_death_rate_calculation_l3216_321659


namespace NUMINAMATH_CALUDE_square_to_eight_acute_triangles_l3216_321675

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive : a > 0 ∧ b > 0 ∧ c > 0
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define an acute-angled triangle
def IsAcuteAngled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 > t.c^2 ∧
  t.b^2 + t.c^2 > t.a^2 ∧
  t.c^2 + t.a^2 > t.b^2

-- Theorem: A square can be divided into 8 acute-angled triangles
theorem square_to_eight_acute_triangles (s : Square) :
  ∃ (t₁ t₂ t₃ t₄ t₅ t₆ t₇ t₈ : Triangle),
    IsAcuteAngled t₁ ∧
    IsAcuteAngled t₂ ∧
    IsAcuteAngled t₃ ∧
    IsAcuteAngled t₄ ∧
    IsAcuteAngled t₅ ∧
    IsAcuteAngled t₆ ∧
    IsAcuteAngled t₇ ∧
    IsAcuteAngled t₈ :=
  sorry

end NUMINAMATH_CALUDE_square_to_eight_acute_triangles_l3216_321675


namespace NUMINAMATH_CALUDE_expression_evaluation_l3216_321656

theorem expression_evaluation (b : ℝ) (h : b = -3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b = 10 / 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3216_321656


namespace NUMINAMATH_CALUDE_curve_crosses_at_2_3_l3216_321685

/-- A curve defined by x = t^2 - 4 and y = t^3 - 6t + 3 for all real t -/
def curve (t : ℝ) : ℝ × ℝ :=
  (t^2 - 4, t^3 - 6*t + 3)

/-- The point where the curve crosses itself -/
def crossing_point : ℝ × ℝ := (2, 3)

/-- Theorem stating that the curve crosses itself at (2, 3) -/
theorem curve_crosses_at_2_3 :
  ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve a = crossing_point :=
sorry

end NUMINAMATH_CALUDE_curve_crosses_at_2_3_l3216_321685


namespace NUMINAMATH_CALUDE_input_statement_incorrect_l3216_321606

-- Define a type for program statements
inductive ProgramStatement
| Input (prompt : String) (value : String)
| Print (prompt : String) (value : String)
| Assignment (left : String) (right : String)

-- Define a function to check if an input statement is valid
def isValidInputStatement (stmt : ProgramStatement) : Prop :=
  match stmt with
  | ProgramStatement.Input _ value => ¬ (value.contains '+' ∨ value.contains '-' ∨ value.contains '*' ∨ value.contains '/')
  | _ => True

-- Theorem to prove
theorem input_statement_incorrect :
  let stmt := ProgramStatement.Input "MATH=" "a+b+c"
  ¬ (isValidInputStatement stmt) := by
sorry

end NUMINAMATH_CALUDE_input_statement_incorrect_l3216_321606


namespace NUMINAMATH_CALUDE_amy_spelling_problems_l3216_321634

/-- The number of spelling problems Amy had to solve -/
def spelling_problems (total_problems math_problems : ℕ) : ℕ :=
  total_problems - math_problems

/-- Proof that Amy had 6 spelling problems -/
theorem amy_spelling_problems :
  spelling_problems 24 18 = 6 := by
  sorry

end NUMINAMATH_CALUDE_amy_spelling_problems_l3216_321634


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3216_321655

/-- Given that p and r · q are inversely proportional, prove that p = 128/15 when q = 10 and r = 3,
    given that p = 16 when q = 8 and r = 2 -/
theorem inverse_proportion_problem (p q r : ℝ) (h1 : ∃ k, p * (r * q) = k) 
  (h2 : p = 16 ∧ q = 8 ∧ r = 2) : 
  (q = 10 ∧ r = 3) → p = 128 / 15 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3216_321655


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_l3216_321617

theorem fraction_meaningful_iff (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x + 3)) ↔ x ≠ -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_l3216_321617


namespace NUMINAMATH_CALUDE_union_intersection_relation_l3216_321693

theorem union_intersection_relation (M N : Set α) : 
  (∃ (x : α), x ∈ M ∩ N → x ∈ M ∪ N) ∧ 
  (∃ (M N : Set α), (∃ (x : α), x ∈ M ∪ N) ∧ M ∩ N = ∅) :=
by sorry

end NUMINAMATH_CALUDE_union_intersection_relation_l3216_321693


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l3216_321666

/-- A quadrilateral with vertices at (1,2), (4,5), (5,4), and (4,1) has a perimeter of 4√2 + 2√10 -/
theorem quadrilateral_perimeter : 
  let vertices : List (ℝ × ℝ) := [(1, 2), (4, 5), (5, 4), (4, 1)]
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let perimeter := (List.zip vertices (vertices.rotateLeft 1)).map (fun (p, q) => distance p q) |>.sum
  perimeter = 4 * Real.sqrt 2 + 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l3216_321666


namespace NUMINAMATH_CALUDE_oil_truck_tank_radius_l3216_321639

/-- Represents a right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ
  height : ℝ

/-- The problem statement -/
theorem oil_truck_tank_radius 
  (stationary_tank : RightCircularCylinder)
  (oil_truck_tank : RightCircularCylinder)
  (oil_level_drop : ℝ)
  (h_stationary_radius : stationary_tank.radius = 100)
  (h_stationary_height : stationary_tank.height = 25)
  (h_truck_height : oil_truck_tank.height = 10)
  (h_oil_drop : oil_level_drop = 0.025)
  (h_volume_equality : π * stationary_tank.radius^2 * oil_level_drop = 
                       π * oil_truck_tank.radius^2 * oil_truck_tank.height) :
  oil_truck_tank.radius = 5 := by
  sorry

#check oil_truck_tank_radius

end NUMINAMATH_CALUDE_oil_truck_tank_radius_l3216_321639


namespace NUMINAMATH_CALUDE_storage_unit_capacity_l3216_321650

/-- A storage unit with three shelves for storing CDs. -/
structure StorageUnit where
  shelf1_racks : ℕ
  shelf1_cds_per_rack : ℕ
  shelf2_racks : ℕ
  shelf2_cds_per_rack : ℕ
  shelf3_racks : ℕ
  shelf3_cds_per_rack : ℕ

/-- Calculate the total number of CDs that can fit in a storage unit. -/
def totalCDs (unit : StorageUnit) : ℕ :=
  unit.shelf1_racks * unit.shelf1_cds_per_rack +
  unit.shelf2_racks * unit.shelf2_cds_per_rack +
  unit.shelf3_racks * unit.shelf3_cds_per_rack

/-- Theorem stating that the specific storage unit can hold 116 CDs. -/
theorem storage_unit_capacity :
  let unit : StorageUnit := {
    shelf1_racks := 5,
    shelf1_cds_per_rack := 8,
    shelf2_racks := 4,
    shelf2_cds_per_rack := 10,
    shelf3_racks := 3,
    shelf3_cds_per_rack := 12
  }
  totalCDs unit = 116 := by
  sorry

end NUMINAMATH_CALUDE_storage_unit_capacity_l3216_321650


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l3216_321678

theorem lcm_hcf_problem (a b : ℕ+) (h1 : a = 8) (h2 : Nat.lcm a b = 24) (h3 : Nat.gcd a b = 4) : b = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l3216_321678


namespace NUMINAMATH_CALUDE_lcm_of_72_108_2100_l3216_321680

theorem lcm_of_72_108_2100 : Nat.lcm (Nat.lcm 72 108) 2100 = 37800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_72_108_2100_l3216_321680


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3216_321668

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -x₁ ∧ y₂ = -y₁

/-- Given that point A(1,a) and point B(b,-2) are symmetric with respect to the origin O, prove that a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin 1 a b (-2)) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3216_321668


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3216_321631

theorem polynomial_factorization 
  (P Q R : Polynomial ℝ) 
  (h : P^4 + Q^4 = R^2) : 
  ∃ (p q r : ℝ) (S : Polynomial ℝ), 
    P = p • S ∧ Q = q • S ∧ R = r • S^2 :=
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3216_321631


namespace NUMINAMATH_CALUDE_ab_ratio_for_inscribed_triangle_l3216_321690

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - p.y)^2 / e.a^2 + (p.x + p.y)^2 / e.b^2 = 1

/-- Checks if two points form a line parallel to y = x -/
def isParallelToYEqualX (p1 p2 : Point) : Prop :=
  p1.x - p1.y = p2.x - p2.y

/-- Theorem: AB/b ratio for an equilateral triangle inscribed in a specific ellipse -/
theorem ab_ratio_for_inscribed_triangle
  (e : Ellipse)
  (t : EquilateralTriangle)
  (h1 : t.A = ⟨0, e.b⟩)
  (h2 : isOnEllipse t.A e ∧ isOnEllipse t.B e ∧ isOnEllipse t.C e)
  (h3 : isParallelToYEqualX t.B t.C)
  (h4 : e.a = e.b * Real.sqrt 2)  -- Condition for focus at vertex C
  : Real.sqrt ((t.B.x - t.A.x)^2 + (t.B.y - t.A.y)^2) / e.b = 8/5 :=
sorry

end NUMINAMATH_CALUDE_ab_ratio_for_inscribed_triangle_l3216_321690


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l3216_321643

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def interior_intersection_points (n : ℕ) : ℕ :=
  Nat.choose n 4

/-- Theorem: The number of distinct interior points where two or more diagonals 
    intersect in a regular decagon is equal to C(10,4) -/
theorem decagon_diagonal_intersections : 
  interior_intersection_points 10 = 210 := by
  sorry

#eval interior_intersection_points 10

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l3216_321643


namespace NUMINAMATH_CALUDE_events_A_B_mutually_exclusive_events_A_C_not_independent_l3216_321696

/-- Represents the possible outcomes when drawing a ball from Box A -/
inductive BoxA
| one
| two
| three
| four

/-- Represents the possible outcomes when drawing a ball from Box B -/
inductive BoxB
| five
| six
| seven
| eight

/-- The type of all possible outcomes when drawing one ball from each box -/
def Outcome := BoxA × BoxB

/-- The sum of the numbers on the balls drawn -/
def sum (o : Outcome) : ℕ :=
  match o with
  | (BoxA.one, b) => 1 + boxBToNat b
  | (BoxA.two, b) => 2 + boxBToNat b
  | (BoxA.three, b) => 3 + boxBToNat b
  | (BoxA.four, b) => 4 + boxBToNat b
where
  boxBToNat : BoxB → ℕ
  | BoxB.five => 5
  | BoxB.six => 6
  | BoxB.seven => 7
  | BoxB.eight => 8

/-- Event A: the sum of the numbers drawn is even -/
def eventA (o : Outcome) : Prop := Even (sum o)

/-- Event B: the sum of the numbers drawn is 9 -/
def eventB (o : Outcome) : Prop := sum o = 9

/-- Event C: the sum of the numbers drawn is greater than 9 -/
def eventC (o : Outcome) : Prop := sum o > 9

/-- The probability measure on the sample space -/
def P : Set Outcome → ℝ := sorry

theorem events_A_B_mutually_exclusive :
  ∀ o : Outcome, ¬(eventA o ∧ eventB o) := by sorry

theorem events_A_C_not_independent :
  P {o | eventA o ∧ eventC o} ≠ P {o | eventA o} * P {o | eventC o} := by sorry

end NUMINAMATH_CALUDE_events_A_B_mutually_exclusive_events_A_C_not_independent_l3216_321696


namespace NUMINAMATH_CALUDE_systematic_sample_validity_l3216_321609

def isValidSystematicSample (sample : List Nat) (populationSize : Nat) (sampleSize : Nat) : Prop :=
  sample.length = sampleSize ∧
  sample.all (· ≤ populationSize) ∧
  sample.all (· > 0) ∧
  ∃ k : Nat, k > 0 ∧ List.zipWith (·-·) (sample.tail) sample = List.replicate (sampleSize - 1) k

theorem systematic_sample_validity :
  isValidSystematicSample [3, 13, 23, 33, 43] 50 5 :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_validity_l3216_321609


namespace NUMINAMATH_CALUDE_darwin_money_problem_l3216_321695

theorem darwin_money_problem (initial_money : ℝ) : 
  (3/4 * (2/3 * initial_money) = 300) → initial_money = 600 := by
  sorry

end NUMINAMATH_CALUDE_darwin_money_problem_l3216_321695


namespace NUMINAMATH_CALUDE_total_sum_calculation_l3216_321612

/-- 
Given that Maggie's share is 75% of the total sum and equals $4,500, 
prove that the total sum is $6,000.
-/
theorem total_sum_calculation (maggies_share : ℝ) (total_sum : ℝ) : 
  maggies_share = 4500 ∧ 
  maggies_share = 0.75 * total_sum →
  total_sum = 6000 := by
  sorry

end NUMINAMATH_CALUDE_total_sum_calculation_l3216_321612


namespace NUMINAMATH_CALUDE_dog_arrangement_theorem_l3216_321670

theorem dog_arrangement_theorem (n : ℕ) (h : n = 5) :
  (n! / 2) = 60 :=
sorry

end NUMINAMATH_CALUDE_dog_arrangement_theorem_l3216_321670


namespace NUMINAMATH_CALUDE_identity_is_unique_solution_l3216_321636

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- The theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f → (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_identity_is_unique_solution_l3216_321636


namespace NUMINAMATH_CALUDE_side_length_of_octagon_l3216_321694

theorem side_length_of_octagon (perimeter : ℝ) (num_sides : ℕ) (h1 : perimeter = 23.6) (h2 : num_sides = 8) :
  perimeter / num_sides = 2.95 := by
  sorry

end NUMINAMATH_CALUDE_side_length_of_octagon_l3216_321694


namespace NUMINAMATH_CALUDE_gray_trees_count_l3216_321614

/-- Represents a drone photograph of an area --/
structure Photograph where
  visible_trees : ℕ
  total_trees : ℕ

/-- Represents a set of three drone photographs of the same area --/
structure PhotoSet where
  photo1 : Photograph
  photo2 : Photograph
  photo3 : Photograph
  equal_total : photo1.total_trees = photo2.total_trees ∧ photo2.total_trees = photo3.total_trees

/-- Calculates the number of trees in gray areas given a set of three photographs --/
def gray_trees (photos : PhotoSet) : ℕ :=
  (photos.photo1.total_trees - photos.photo1.visible_trees) +
  (photos.photo2.total_trees - photos.photo2.visible_trees)

/-- Theorem stating that for the given set of photographs, the number of trees in gray areas is 26 --/
theorem gray_trees_count (photos : PhotoSet)
  (h1 : photos.photo1.visible_trees = 100)
  (h2 : photos.photo2.visible_trees = 90)
  (h3 : photos.photo3.visible_trees = 82) :
  gray_trees photos = 26 := by
  sorry


end NUMINAMATH_CALUDE_gray_trees_count_l3216_321614


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3216_321654

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = -6 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 1 → 
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3216_321654


namespace NUMINAMATH_CALUDE_brown_dogs_count_l3216_321644

/-- Represents the number of dogs in a kennel with specific characteristics. -/
structure DogKennel where
  total : ℕ
  longFur : ℕ
  neitherLongFurNorBrown : ℕ
  longFurAndBrown : ℕ

/-- Theorem stating the number of brown dogs in the kennel. -/
theorem brown_dogs_count (k : DogKennel)
    (h1 : k.total = 45)
    (h2 : k.longFur = 29)
    (h3 : k.neitherLongFurNorBrown = 8)
    (h4 : k.longFurAndBrown = 9) :
    k.total - k.neitherLongFurNorBrown - (k.longFur - k.longFurAndBrown) = 17 := by
  sorry

#check brown_dogs_count

end NUMINAMATH_CALUDE_brown_dogs_count_l3216_321644


namespace NUMINAMATH_CALUDE_quadratic_discriminant_value_l3216_321665

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_discriminant_value (a : ℝ) :
  discriminant 1 (-3) (-2*a) = 1 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_value_l3216_321665


namespace NUMINAMATH_CALUDE_students_not_in_program_x_l3216_321688

/-- Represents a grade level in the school -/
inductive GradeLevel
  | Elementary
  | Middle
  | High

/-- Represents the gender of students -/
inductive Gender
  | Girl
  | Boy

/-- The number of students in each grade level and gender -/
def studentCount (level : GradeLevel) (gender : Gender) : ℕ :=
  match level, gender with
  | GradeLevel.Elementary, Gender.Girl => 192
  | GradeLevel.Elementary, Gender.Boy => 135
  | GradeLevel.Middle, Gender.Girl => 233
  | GradeLevel.Middle, Gender.Boy => 163
  | GradeLevel.High, Gender.Girl => 117
  | GradeLevel.High, Gender.Boy => 89

/-- The number of students in Program X for each grade level and gender -/
def programXCount (level : GradeLevel) (gender : Gender) : ℕ :=
  match level, gender with
  | GradeLevel.Elementary, Gender.Girl => 48
  | GradeLevel.Elementary, Gender.Boy => 28
  | GradeLevel.Middle, Gender.Girl => 98
  | GradeLevel.Middle, Gender.Boy => 51
  | GradeLevel.High, Gender.Girl => 40
  | GradeLevel.High, Gender.Boy => 25

/-- The total number of students not participating in Program X -/
def studentsNotInProgramX : ℕ :=
  (studentCount GradeLevel.Elementary Gender.Girl - programXCount GradeLevel.Elementary Gender.Girl) +
  (studentCount GradeLevel.Elementary Gender.Boy - programXCount GradeLevel.Elementary Gender.Boy) +
  (studentCount GradeLevel.Middle Gender.Girl - programXCount GradeLevel.Middle Gender.Girl) +
  (studentCount GradeLevel.Middle Gender.Boy - programXCount GradeLevel.Middle Gender.Boy) +
  (studentCount GradeLevel.High Gender.Girl - programXCount GradeLevel.High Gender.Girl) +
  (studentCount GradeLevel.High Gender.Boy - programXCount GradeLevel.High Gender.Boy)

theorem students_not_in_program_x :
  studentsNotInProgramX = 639 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_program_x_l3216_321688


namespace NUMINAMATH_CALUDE_triangle_problem_l3216_321674

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.c = 3)
  (h3 : Real.cos t.B = 1/4) :
  t.b = Real.sqrt 10 ∧ Real.sin (2 * t.C) = (3 * Real.sqrt 15) / 16 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3216_321674


namespace NUMINAMATH_CALUDE_manolo_face_mask_production_l3216_321648

/-- Represents the face-mask production scenario for Manolo -/
structure FaceMaskProduction where
  initial_rate : ℕ  -- Rate of production in the first hour (minutes per mask)
  total_masks : ℕ   -- Total masks produced in a 4-hour shift
  shift_duration : ℕ -- Total duration of the shift in hours

/-- Calculates the time required to make one face-mask after the first hour -/
def time_per_mask_after_first_hour (p : FaceMaskProduction) : ℕ :=
  let masks_in_first_hour := 60 / p.initial_rate
  let remaining_masks := p.total_masks - masks_in_first_hour
  let remaining_time := (p.shift_duration - 1) * 60
  remaining_time / remaining_masks

/-- Theorem stating that given the initial conditions, the time per mask after the first hour is 6 minutes -/
theorem manolo_face_mask_production :
  ∀ (p : FaceMaskProduction),
    p.initial_rate = 4 ∧
    p.total_masks = 45 ∧
    p.shift_duration = 4 →
    time_per_mask_after_first_hour p = 6 := by
  sorry

end NUMINAMATH_CALUDE_manolo_face_mask_production_l3216_321648


namespace NUMINAMATH_CALUDE_expected_rolls_for_2010_l3216_321683

/-- Represents the probability of getting a certain sum with a fair six-sided die -/
def probability (n : ℕ) : ℚ :=
  sorry

/-- Represents the expected number of rolls to reach a sum of n with a fair six-sided die -/
def expected_rolls (n : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the expected number of rolls to reach a sum of 2010 -/
theorem expected_rolls_for_2010 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000000 ∧ 
  abs (expected_rolls 2010 - 574523809/1000000) < ε :=
sorry

end NUMINAMATH_CALUDE_expected_rolls_for_2010_l3216_321683


namespace NUMINAMATH_CALUDE_solution_check_l3216_321673

theorem solution_check (x : ℝ) : x = 2 →
  (2 * x - 4 = 0) ∧ 
  (3 * x + 6 ≠ 0) ∧ 
  (2 * x + 4 ≠ 0) ∧ 
  (1/2 * x ≠ -4) := by
sorry

end NUMINAMATH_CALUDE_solution_check_l3216_321673


namespace NUMINAMATH_CALUDE_f_increasing_after_3_l3216_321622

def f (x : ℝ) := 2 * (x - 3)^2 - 1

theorem f_increasing_after_3 :
  ∀ x₁ x₂ : ℝ, x₁ ≥ 3 → x₂ ≥ 3 → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_after_3_l3216_321622


namespace NUMINAMATH_CALUDE_rectangle_opposite_sides_equal_square_all_sides_equal_l3216_321603

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a square
structure Square where
  side : ℝ

-- Theorem for rectangle
theorem rectangle_opposite_sides_equal (r : Rectangle) : 
  r.width = r.width ∧ r.height = r.height := by
  sorry

-- Theorem for square
theorem square_all_sides_equal (s : Square) : 
  s.side = s.side ∧ s.side = s.side ∧ s.side = s.side ∧ s.side = s.side := by
  sorry

end NUMINAMATH_CALUDE_rectangle_opposite_sides_equal_square_all_sides_equal_l3216_321603


namespace NUMINAMATH_CALUDE_green_blue_difference_l3216_321698

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  color_sum : blue + yellow + green = total
  ratio : blue * 18 = total * 3 ∧ yellow * 18 = total * 7 ∧ green * 18 = total * 8

theorem green_blue_difference (bag : DiskBag) (h : bag.total = 72) : 
  bag.green - bag.blue = 20 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_l3216_321698


namespace NUMINAMATH_CALUDE_B_2_2_equals_12_l3216_321657

def B : ℕ → ℕ → ℕ
| 0, n => n + 2
| m + 1, 0 => B m 2
| m + 1, n + 1 => B m (B (m + 1) n)

theorem B_2_2_equals_12 : B 2 2 = 12 := by sorry

end NUMINAMATH_CALUDE_B_2_2_equals_12_l3216_321657
