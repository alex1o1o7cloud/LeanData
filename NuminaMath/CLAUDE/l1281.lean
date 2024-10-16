import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_exterior_120_is_equilateral_equal_angles_is_equilateral_two_angles_70_40_is_isosceles_l1281_128150

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  sum_angles : angle_a + angle_b + angle_c = 180

-- Define an isosceles triangle
def IsoscelesTriangle (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Define an equilateral triangle
def EquilateralTriangle (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Statement 1
theorem isosceles_exterior_120_is_equilateral (t : Triangle) (h : IsoscelesTriangle t) :
  ∃ (ext_angle : ℝ), ext_angle = 120 → EquilateralTriangle t :=
sorry

-- Statement 2
theorem equal_angles_is_equilateral (t : Triangle) :
  t.angle_a = t.angle_b ∧ t.angle_b = t.angle_c → EquilateralTriangle t :=
sorry

-- Statement 3
theorem two_angles_70_40_is_isosceles (t : Triangle) :
  t.angle_a = 70 ∧ t.angle_b = 40 → IsoscelesTriangle t :=
sorry

end NUMINAMATH_CALUDE_isosceles_exterior_120_is_equilateral_equal_angles_is_equilateral_two_angles_70_40_is_isosceles_l1281_128150


namespace NUMINAMATH_CALUDE_total_wheels_in_lot_l1281_128107

/-- The number of wheels on a standard car -/
def wheels_per_car : ℕ := 4

/-- The number of cars in the parking lot -/
def cars_in_lot : ℕ := 17

/-- Theorem: The total number of car wheels in the parking lot is 68 -/
theorem total_wheels_in_lot : cars_in_lot * wheels_per_car = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_in_lot_l1281_128107


namespace NUMINAMATH_CALUDE_crossword_puzzle_subset_l1281_128136

def is_three_identical_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ ∃ d, n = d * 100 + d * 10 + d

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def has_three_middle_threes (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ ∃ a b, n = a * 10000 + 3 * 1000 + 3 * 100 + 3 * 10 + b

theorem crossword_puzzle_subset :
  ∀ x y z : ℕ,
  is_three_identical_digits x →
  y = x^2 →
  digit_sum z = 18 →
  has_three_middle_threes z →
  x = 111 ∧ y = 12321 ∧ z = 33333 :=
by sorry

end NUMINAMATH_CALUDE_crossword_puzzle_subset_l1281_128136


namespace NUMINAMATH_CALUDE_bubble_sort_probability_l1281_128137

theorem bubble_sort_probability (n : ℕ) (h : n = 36) :
  let arrangements := n.factorial
  let favorable_outcomes := (n - 2).factorial
  (favorable_outcomes : ℚ) / arrangements = 1 / 1260 := by
  sorry

end NUMINAMATH_CALUDE_bubble_sort_probability_l1281_128137


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1281_128164

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 4 / (x + y)^2 ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 4 / (x + y)^2 = 2 * Real.sqrt 2 ↔ x = y ∧ x = (Real.sqrt 2)^(3/4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1281_128164


namespace NUMINAMATH_CALUDE_student_number_problem_l1281_128149

theorem student_number_problem (x : ℝ) : 6 * x - 138 = 102 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1281_128149


namespace NUMINAMATH_CALUDE_fraction_inequality_l1281_128159

theorem fraction_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hca : c > a) (hab : a > b) : 
  a / (c - a) > b / (c - b) := by
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1281_128159


namespace NUMINAMATH_CALUDE_det_A_eq_90_l1281_128144

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![3, 0, 2],
    ![8, 5, -2],
    ![3, 3, 6]]

theorem det_A_eq_90 : Matrix.det A = 90 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_90_l1281_128144


namespace NUMINAMATH_CALUDE_fraction_power_equality_l1281_128181

theorem fraction_power_equality : (72000 ^ 4 : ℝ) / (24000 ^ 4) = 81 := by sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l1281_128181


namespace NUMINAMATH_CALUDE_exactly_two_clubs_l1281_128132

theorem exactly_two_clubs (S : ℕ) (A B C ABC : ℕ) : 
  S = 400 ∧
  A = S / 2 ∧
  B = S * 5 / 8 ∧
  C = S * 3 / 4 ∧
  ABC = S * 3 / 8 ∧
  A + B + C - 2 * ABC ≥ S →
  A + B + C - S - ABC = 500 := by
sorry

end NUMINAMATH_CALUDE_exactly_two_clubs_l1281_128132


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l1281_128168

def arrange_books (math_books : ℕ) (english_books : ℕ) : ℕ :=
  let math_group := 1
  let english_groups := 2
  let total_groups := math_group + english_groups
  let math_arrangements := Nat.factorial math_books
  let english_group_size := english_books / english_groups
  let english_group_arrangements := Nat.factorial english_group_size
  Nat.factorial total_groups * math_arrangements * english_group_arrangements * english_group_arrangements

theorem book_arrangement_theorem :
  arrange_books 4 6 = 5184 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l1281_128168


namespace NUMINAMATH_CALUDE_parallel_line_point_slope_form_l1281_128188

/-- Given points A, B, and C in the plane, this theorem states that the line passing through A
    and parallel to BC has the specified point-slope form. -/
theorem parallel_line_point_slope_form 
  (A B C : ℝ × ℝ) 
  (hA : A = (4, 6)) 
  (hB : B = (-3, -1)) 
  (hC : C = (5, -5)) : 
  ∃ (m : ℝ), m = -1/2 ∧ 
  ∀ (x y : ℝ), (y - 6 = m * (x - 4) ↔ 
    (∃ (t : ℝ), (x, y) = A + t • (C - B) ∧ (x, y) ≠ A)) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_point_slope_form_l1281_128188


namespace NUMINAMATH_CALUDE_ellipse_and_trajectory_l1281_128186

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Point A is on the ellipse C -/
def point_A_on_C (a b : ℝ) : Prop := ellipse_C 1 (3/2) a b

/-- Sum of distances from A to foci equals 4 -/
def sum_distances_4 (a : ℝ) : Prop := 2 * a = 4

/-- Conditions on a and b -/
def a_b_conditions (a b : ℝ) : Prop := a > b ∧ b > 0

/-- Theorem stating the equation of ellipse C and the trajectory of midpoint M -/
theorem ellipse_and_trajectory (a b : ℝ) 
  (h1 : a_b_conditions a b) 
  (h2 : point_A_on_C a b) 
  (h3 : sum_distances_4 a) : 
  (∀ x y, ellipse_C x y a b ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ F1 : ℝ × ℝ, ∀ x y, 
    (∃ x1 y1, ellipse_C x1 y1 a b ∧ x = (F1.1 + x1) / 2 ∧ y = (F1.2 + y1) / 2) ↔ 
    (x + 1/2)^2 + 4*y^2 / 3 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_trajectory_l1281_128186


namespace NUMINAMATH_CALUDE_test_score_probability_and_expectation_l1281_128195

-- Define the scoring system
def score_correct : ℕ := 5
def score_incorrect : ℕ := 0

-- Define the total number of questions and correct answers
def total_questions : ℕ := 10
def correct_answers : ℕ := 6

-- Define the probabilities for the remaining questions
def prob_two_eliminated : ℚ := 1/2
def prob_one_eliminated : ℚ := 1/3
def prob_guessed : ℚ := 1/4

-- Define the score distribution
def score_distribution : List (ℕ × ℚ) := [
  (30, 1/8),
  (35, 17/48),
  (40, 17/48),
  (45, 7/48),
  (50, 1/48)
]

-- Theorem statement
theorem test_score_probability_and_expectation :
  (List.lookup 45 score_distribution = some (7/48)) ∧
  (List.foldl (λ acc (score, prob) => acc + score * prob) 0 score_distribution = 455/12) := by
  sorry

end NUMINAMATH_CALUDE_test_score_probability_and_expectation_l1281_128195


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l1281_128100

theorem max_value_cos_sin (θ : Real) (h : 0 < θ ∧ θ < π) :
  ∃ (M : Real), M = (3 : Real) / 2 ∧
  ∀ φ, 0 < φ ∧ φ < π →
    Real.cos (φ / 2) * (2 - Real.sin φ) ≤ M ∧
    ∃ ψ, 0 < ψ ∧ ψ < π ∧ Real.cos (ψ / 2) * (2 - Real.sin ψ) = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l1281_128100


namespace NUMINAMATH_CALUDE_absolute_value_inequality_find_a_value_l1281_128148

-- Part 1
theorem absolute_value_inequality (x : ℝ) :
  (|x - 1| + |x + 2| ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) := by sorry

-- Part 2
theorem find_a_value (a : ℝ) :
  (∀ x, |a*x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3) → a = -3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_find_a_value_l1281_128148


namespace NUMINAMATH_CALUDE_optimal_garden_length_l1281_128160

/-- Represents the length of the side perpendicular to the greenhouse -/
def x : ℝ := sorry

/-- The total amount of fencing available -/
def total_fence : ℝ := 280

/-- The maximum allowed length of the side parallel to the greenhouse -/
def max_parallel_length : ℝ := 300

/-- The length of the side parallel to the greenhouse -/
def parallel_length (x : ℝ) : ℝ := total_fence - 2 * x

/-- The area of the garden as a function of x -/
def garden_area (x : ℝ) : ℝ := x * (parallel_length x)

/-- Theorem stating that the optimal length of the side parallel to the greenhouse is 140 feet -/
theorem optimal_garden_length :
  ∃ (x : ℝ), 
    x > 0 ∧ 
    parallel_length x ≤ max_parallel_length ∧ 
    parallel_length x = 140 ∧ 
    ∀ (y : ℝ), y > 0 ∧ parallel_length y ≤ max_parallel_length → 
      garden_area x ≥ garden_area y :=
by sorry

end NUMINAMATH_CALUDE_optimal_garden_length_l1281_128160


namespace NUMINAMATH_CALUDE_lcm_theorem_l1281_128176

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def lcm_condition (ab cd : ℕ) : Prop :=
  is_two_digit ab ∧ is_two_digit cd ∧
  Nat.lcm ab cd = (7 * Nat.lcm (reverse_digits ab) (reverse_digits cd)) / 4

theorem lcm_theorem (ab cd : ℕ) (h : lcm_condition ab cd) :
  Nat.lcm ab cd = 252 := by
  sorry

end NUMINAMATH_CALUDE_lcm_theorem_l1281_128176


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_range_l1281_128143

theorem quadratic_always_positive_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_range_l1281_128143


namespace NUMINAMATH_CALUDE_bank_coin_count_l1281_128167

/-- The total number of coins turned in by a customer at a bank -/
def total_coins (dimes nickels quarters : ℕ) : ℕ :=
  dimes + nickels + quarters

/-- Theorem stating that the total number of coins is 11 given the specific quantities -/
theorem bank_coin_count : total_coins 2 2 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bank_coin_count_l1281_128167


namespace NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1281_128192

/-- The number of rectangles in a row of length n -/
def rectangles_in_row (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of rectangles in an m × n grid -/
def rectangles_in_grid (m n : ℕ) : ℕ :=
  m * n + m * rectangles_in_row n + n * rectangles_in_row m

theorem rectangles_in_5x4_grid :
  rectangles_in_grid 5 4 = 24 := by sorry

end NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1281_128192


namespace NUMINAMATH_CALUDE_abs_x_minus_one_equals_one_minus_x_implies_x_leq_one_l1281_128126

theorem abs_x_minus_one_equals_one_minus_x_implies_x_leq_one (x : ℝ) : 
  |x - 1| = 1 - x → x ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_equals_one_minus_x_implies_x_leq_one_l1281_128126


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1281_128189

theorem quadratic_function_properties (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^2 + b * x + c
  (f (-2) = 0) → (f 3 = 0) → (f (-b / (2 * a)) > 0) →
  (a < 0) ∧ 
  ({x : ℝ | a * x + c > 0} = {x : ℝ | x > 6}) ∧
  (a + b + c > 0) ∧
  ({x : ℝ | c * x^2 - b * x + a < 0} = {x : ℝ | -1/3 < x ∧ x < 1/2}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1281_128189


namespace NUMINAMATH_CALUDE_time_period_is_three_years_l1281_128171

/-- Calculates the time period for which a sum is due given the banker's gain, banker's discount, and interest rate. -/
def calculate_time_period (bankers_gain : ℚ) (bankers_discount : ℚ) (interest_rate : ℚ) : ℚ :=
  let true_discount := bankers_discount - bankers_gain
  let ratio := bankers_discount / true_discount
  (ratio - 1) / (interest_rate / 100)

/-- Theorem stating that given the specific values in the problem, the time period is 3 years. -/
theorem time_period_is_three_years :
  let bankers_gain : ℚ := 90
  let bankers_discount : ℚ := 340
  let interest_rate : ℚ := 12
  calculate_time_period bankers_gain bankers_discount interest_rate = 3 := by
  sorry

#eval calculate_time_period 90 340 12

end NUMINAMATH_CALUDE_time_period_is_three_years_l1281_128171


namespace NUMINAMATH_CALUDE_kelly_games_left_l1281_128103

theorem kelly_games_left (initial_games give_away_games : ℕ) 
  (h1 : initial_games = 257)
  (h2 : give_away_games = 138) :
  initial_games - give_away_games = 119 :=
by sorry

end NUMINAMATH_CALUDE_kelly_games_left_l1281_128103


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_plus_reciprocal_l1281_128104

theorem imaginary_part_of_z_plus_reciprocal (z : ℂ) (h : z = 1 + I) :
  (z + z⁻¹).im = 1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_plus_reciprocal_l1281_128104


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_three_l1281_128112

theorem trigonometric_product_equals_three :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_three_l1281_128112


namespace NUMINAMATH_CALUDE_angle_sum_equation_l1281_128191

open Real

theorem angle_sum_equation (x : ℝ) : 
  (0 ≤ x ∧ x ≤ 2 * π) →
  (cos x)^5 - (sin x)^5 = 1 / sin x - 1 / cos x →
  ∃ (y : ℝ), (0 ≤ y ∧ y ≤ 2 * π) ∧
             (cos y)^5 - (sin y)^5 = 1 / sin y - 1 / cos y ∧
             x + y = 3 * π / 2 :=
by sorry

end NUMINAMATH_CALUDE_angle_sum_equation_l1281_128191


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l1281_128138

theorem prime_sum_theorem (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p + q = r → p < q → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l1281_128138


namespace NUMINAMATH_CALUDE_peter_hunts_triple_mark_l1281_128179

/-- The number of animals hunted by each person in a day --/
structure HuntingData where
  sam : ℕ
  rob : ℕ
  mark : ℕ
  peter : ℕ

/-- The conditions of the hunting problem --/
def huntingProblem (h : HuntingData) : Prop :=
  h.sam = 6 ∧
  h.rob = h.sam / 2 ∧
  h.mark = (h.sam + h.rob) / 3 ∧
  h.sam + h.rob + h.mark + h.peter = 21

/-- The theorem stating that Peter hunts 3 times more animals than Mark --/
theorem peter_hunts_triple_mark (h : HuntingData) 
  (hcond : huntingProblem h) : h.peter = 3 * h.mark := by
  sorry

end NUMINAMATH_CALUDE_peter_hunts_triple_mark_l1281_128179


namespace NUMINAMATH_CALUDE_a_2_times_a_3_l1281_128166

def a (n : ℕ) : ℕ :=
  if n % 2 = 1 then 3 * n + 1 else 2 * n - 2

theorem a_2_times_a_3 : a 2 * a 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_a_2_times_a_3_l1281_128166


namespace NUMINAMATH_CALUDE_bookshelf_discount_percentage_l1281_128183

theorem bookshelf_discount_percentage (discount : ℝ) (final_price : ℝ) (tax_rate : ℝ) : 
  discount = 4.50 →
  final_price = 49.50 →
  tax_rate = 0.10 →
  (discount / (final_price / (1 + tax_rate) + discount)) * 100 = 9 := by
sorry

end NUMINAMATH_CALUDE_bookshelf_discount_percentage_l1281_128183


namespace NUMINAMATH_CALUDE_smallest_c_inequality_l1281_128184

theorem smallest_c_inequality (c : ℝ) : 
  (∀ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 → 
    (x * y * z) ^ (1/3 : ℝ) + c * |x - y + z| ≥ (x + y + z) / 3) ↔ 
  c ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_inequality_l1281_128184


namespace NUMINAMATH_CALUDE_exists_painted_subpolygon_l1281_128185

/-- Represents a convex polygon --/
structure ConvexPolygon where
  -- Add necessary fields

/-- Represents a diagonal of a polygon --/
structure Diagonal where
  -- Add necessary fields

/-- Represents a subpolygon formed by diagonals --/
structure Subpolygon where
  -- Add necessary fields

/-- A function to check if a subpolygon is entirely painted on the outside --/
def is_entirely_painted_outside (sp : Subpolygon) : Prop :=
  sorry

/-- The main theorem --/
theorem exists_painted_subpolygon 
  (P : ConvexPolygon) 
  (sides_painted_outside : Prop) 
  (diagonals : List Diagonal)
  (no_three_intersect : Prop)
  (diagonals_painted_one_side : Prop) :
  ∃ (sp : Subpolygon), is_entirely_painted_outside sp :=
sorry

end NUMINAMATH_CALUDE_exists_painted_subpolygon_l1281_128185


namespace NUMINAMATH_CALUDE_room_width_calculation_l1281_128196

theorem room_width_calculation (area : ℝ) (length : ℝ) (width : ℝ) :
  area = 10 ∧ length = 5 ∧ area = length * width → width = 2 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l1281_128196


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_150_75_l1281_128133

theorem largest_two_digit_prime_factor_of_binom_150_75 :
  (∃ (p : ℕ), Prime p ∧ 10 ≤ p ∧ p < 100 ∧ p ∣ Nat.choose 150 75 ∧
    ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 150 75 → q ≤ p) →
  (73 : ℕ).Prime ∧ 73 ∣ Nat.choose 150 75 ∧
    ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 150 75 → q ≤ 73 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_150_75_l1281_128133


namespace NUMINAMATH_CALUDE_three_power_fraction_equals_five_fourths_l1281_128124

theorem three_power_fraction_equals_five_fourths :
  (3^100 + 3^98) / (3^100 - 3^98) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_three_power_fraction_equals_five_fourths_l1281_128124


namespace NUMINAMATH_CALUDE_meatballs_stolen_l1281_128174

theorem meatballs_stolen (original_total original_beef original_chicken original_pork remaining_beef remaining_chicken remaining_pork : ℕ) :
  original_total = 30 →
  original_beef = 15 →
  original_chicken = 10 →
  original_pork = 5 →
  remaining_beef = 10 →
  remaining_chicken = 10 →
  remaining_pork = 5 →
  original_beef - remaining_beef = 5 :=
by sorry

end NUMINAMATH_CALUDE_meatballs_stolen_l1281_128174


namespace NUMINAMATH_CALUDE_sum_of_digits_divisible_by_11_l1281_128198

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among any 39 consecutive natural numbers, there is always one whose sum of digits is divisible by 11 -/
theorem sum_of_digits_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k ≤ 38 ∧ (sum_of_digits (N + k)) % 11 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_divisible_by_11_l1281_128198


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l1281_128165

/-- Probability of selecting two non-defective pens from a box of pens -/
theorem prob_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (h1 : total_pens = 12) 
  (h2 : defective_pens = 6) : 
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 5 / 22 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l1281_128165


namespace NUMINAMATH_CALUDE_total_sum_lent_l1281_128127

/-- Represents the sum of money lent in two parts -/
structure LoanParts where
  first : ℝ
  second : ℝ

/-- Calculates the interest for a given principal, rate, and time -/
def interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem total_sum_lent (loan : LoanParts) :
  loan.second = 1648 →
  interest loan.first 0.03 8 = interest loan.second 0.05 3 →
  loan.first + loan.second = 2678 := by
  sorry

end NUMINAMATH_CALUDE_total_sum_lent_l1281_128127


namespace NUMINAMATH_CALUDE_double_price_profit_l1281_128145

theorem double_price_profit (cost_price : ℝ) (initial_selling_price : ℝ) :
  initial_selling_price = cost_price * 1.5 →
  let double_price := 2 * initial_selling_price
  (double_price - cost_price) / cost_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_double_price_profit_l1281_128145


namespace NUMINAMATH_CALUDE_internship_arrangements_l1281_128155

/-- The number of ways to arrange 4 distinct objects into 2 indistinguishable pairs 
    and then assign these pairs to 2 distinct locations -/
theorem internship_arrangements (n : Nat) (m : Nat) : n = 4 ∧ m = 2 → 
  (Nat.choose n 2 / 2) * m.factorial = 6 := by
  sorry

end NUMINAMATH_CALUDE_internship_arrangements_l1281_128155


namespace NUMINAMATH_CALUDE_joint_savings_account_total_l1281_128111

def kimmie_earnings : ℚ := 1950
def zahra_earnings_ratio : ℚ := 1 - 2/3
def layla_earnings_ratio : ℚ := 9/4
def kimmie_savings_rate : ℚ := 35/100
def zahra_savings_rate : ℚ := 40/100
def layla_savings_rate : ℚ := 30/100

theorem joint_savings_account_total :
  let zahra_earnings := kimmie_earnings * zahra_earnings_ratio
  let layla_earnings := kimmie_earnings * layla_earnings_ratio
  let kimmie_savings := kimmie_earnings * kimmie_savings_rate
  let zahra_savings := zahra_earnings * zahra_savings_rate
  let layla_savings := layla_earnings * layla_savings_rate
  let total_savings := kimmie_savings + zahra_savings + layla_savings
  total_savings = 2258.75 := by
  sorry

end NUMINAMATH_CALUDE_joint_savings_account_total_l1281_128111


namespace NUMINAMATH_CALUDE_weight_of_B_l1281_128105

/-- Given three weights A, B, and C, prove that B equals 31 when:
    - The average of A, B, and C is 45
    - The average of A and B is 40
    - The average of B and C is 43 -/
theorem weight_of_B (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) : 
  B = 31 := by
  sorry

#check weight_of_B

end NUMINAMATH_CALUDE_weight_of_B_l1281_128105


namespace NUMINAMATH_CALUDE_fib_like_seq_a9_l1281_128110

/-- An increasing sequence of positive integers with a Fibonacci-like recurrence relation -/
def FibLikeSeq (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a (n + 1) + a n)

theorem fib_like_seq_a9 (a : ℕ → ℕ) (h : FibLikeSeq a) (h7 : a 7 = 210) : 
  a 9 = 550 := by
  sorry

end NUMINAMATH_CALUDE_fib_like_seq_a9_l1281_128110


namespace NUMINAMATH_CALUDE_intersection_range_l1281_128123

-- Define the function f(x) = |x^2 - 4x + 3|
def f (x : ℝ) : ℝ := abs (x^2 - 4*x + 3)

-- Define the property of having at least three intersections
def has_at_least_three_intersections (b : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = b ∧ f x₂ = b ∧ f x₃ = b

-- State the theorem
theorem intersection_range :
  ∀ b : ℝ, has_at_least_three_intersections b ↔ 0 < b ∧ b ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_range_l1281_128123


namespace NUMINAMATH_CALUDE_units_digit_of_5_pow_17_times_4_l1281_128102

theorem units_digit_of_5_pow_17_times_4 : ∃ n : ℕ, 5^17 * 4 = 10 * n :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_5_pow_17_times_4_l1281_128102


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l1281_128175

/-- Given two concentric circles with radii R and r, where the area of the ring between them is 20π,
    the length of a chord of the larger circle that is tangent to the smaller circle is 4√5. -/
theorem chord_length_concentric_circles (R r : ℝ) (h : R > r) :
  (π * R^2 - π * r^2 = 20 * π) →
  ∃ (c : ℝ), c^2 = 80 ∧ c = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l1281_128175


namespace NUMINAMATH_CALUDE_women_married_fraction_l1281_128146

theorem women_married_fraction (total : ℕ) (h_total_pos : total > 0) :
  let women := (76 : ℚ) / 100 * total
  let married := (60 : ℚ) / 100 * total
  let men := total - women
  let single_men := (2 : ℚ) / 3 * men
  let married_men := men - single_men
  let married_women := married - married_men
  married_women / women = 13 / 19 := by
sorry

end NUMINAMATH_CALUDE_women_married_fraction_l1281_128146


namespace NUMINAMATH_CALUDE_base_10_to_base_3_172_l1281_128128

/-- Converts a natural number to its base 3 representation as a list of digits -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

theorem base_10_to_base_3_172 :
  toBase3 172 = [2, 0, 1, 0, 1] := by
  sorry

#eval toBase3 172  -- This line is for verification purposes

end NUMINAMATH_CALUDE_base_10_to_base_3_172_l1281_128128


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1281_128152

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 2 * i) / (1 + 4 * i) = -6 / 17 - (10 / 17) * i :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1281_128152


namespace NUMINAMATH_CALUDE_bicycle_distance_l1281_128139

theorem bicycle_distance (front_circ rear_circ : ℚ) (extra_revs : ℕ) : 
  front_circ = 4/3 →
  rear_circ = 3/2 →
  extra_revs = 25 →
  (front_circ * (extra_revs + (rear_circ * extra_revs) / (front_circ - rear_circ))) = 300 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_distance_l1281_128139


namespace NUMINAMATH_CALUDE_q_min_at_2_l1281_128113

-- Define the function q
def q (x : ℝ) : ℝ := (x - 5)^2 + (x - 2)^2 - 6

-- State the theorem
theorem q_min_at_2 : 
  ∀ x : ℝ, q x ≥ q 2 :=
sorry

end NUMINAMATH_CALUDE_q_min_at_2_l1281_128113


namespace NUMINAMATH_CALUDE_shoe_cost_calculation_l1281_128108

def initial_savings : ℕ := 30
def earnings_per_lawn : ℕ := 5
def lawns_per_weekend : ℕ := 3
def weekends_needed : ℕ := 6

def total_earnings : ℕ := earnings_per_lawn * lawns_per_weekend * weekends_needed

theorem shoe_cost_calculation :
  initial_savings + total_earnings = 120 := by sorry

end NUMINAMATH_CALUDE_shoe_cost_calculation_l1281_128108


namespace NUMINAMATH_CALUDE_science_olympiad_participation_l1281_128153

theorem science_olympiad_participation 
  (j s : ℕ) -- j: number of juniors, s: number of seniors
  (h1 : (3 : ℚ) / 7 * j = (2 : ℚ) / 7 * s) -- equal number of participants
  : s = 3 * j := by
sorry

end NUMINAMATH_CALUDE_science_olympiad_participation_l1281_128153


namespace NUMINAMATH_CALUDE_heating_rate_at_10_seconds_l1281_128125

-- Define the temperature function
def temperature (t : ℝ) : ℝ := 0.2 * t^2

-- Define the rate of heating function (derivative of temperature)
def rateOfHeating (t : ℝ) : ℝ := 0.4 * t

-- Theorem statement
theorem heating_rate_at_10_seconds :
  rateOfHeating 10 = 4 := by sorry

end NUMINAMATH_CALUDE_heating_rate_at_10_seconds_l1281_128125


namespace NUMINAMATH_CALUDE_children_age_sum_l1281_128173

theorem children_age_sum :
  let num_children : ℕ := 5
  let age_interval : ℕ := 3
  let youngest_age : ℕ := 6
  let ages : List ℕ := List.range num_children |>.map (fun i => youngest_age + i * age_interval)
  ages.sum = 60 := by
  sorry

end NUMINAMATH_CALUDE_children_age_sum_l1281_128173


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_right_triangle_l1281_128142

theorem circumscribed_circle_area_right_triangle 
  (AC BC : ℝ) (h_right_triangle : AC^2 + BC^2 = (AC + BC)^2 / 2) 
  (h_AC : AC = 6) (h_BC : BC = 8) : 
  let r := ((AC^2 + BC^2).sqrt) / 2
  π * r^2 = 25 * π := by
sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_right_triangle_l1281_128142


namespace NUMINAMATH_CALUDE_geometric_sequence_a1_l1281_128157

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_a1 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  a 3 = 1 →
  a 5 = 1/2 →
  a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a1_l1281_128157


namespace NUMINAMATH_CALUDE_infinite_points_in_circle_l1281_128135

/-- A point in the xy-plane with rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ

/-- Predicate to check if a point is strictly inside the circle -/
def is_inside_circle (p : RationalPoint) : Prop :=
  p.x^2 + p.y^2 < 9

/-- Predicate to check if a point has positive coordinates -/
def has_positive_coordinates (p : RationalPoint) : Prop :=
  p.x > 0 ∧ p.y > 0

theorem infinite_points_in_circle :
  ∃ (S : Set RationalPoint), (∀ p ∈ S, is_inside_circle p ∧ has_positive_coordinates p) ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_infinite_points_in_circle_l1281_128135


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l1281_128156

theorem missing_fraction_sum (x : ℚ) : x = -11/60 →
  1/3 + 1/2 + (-5/6) + 1/5 + 1/4 + (-2/15) + x = 0.13333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l1281_128156


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1281_128182

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 827 * n ≡ 1369 * n [ZMOD 36] ∧ ∀ (m : ℕ), m > 0 → 827 * m ≡ 1369 * m [ZMOD 36] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1281_128182


namespace NUMINAMATH_CALUDE_reach_target_probability_l1281_128116

-- Define the number of lily pads
def num_pads : ℕ := 16

-- Define the starting pad
def start_pad : ℕ := 2

-- Define the target pad
def target_pad : ℕ := 14

-- Define the predator pads
def predator_pads : List ℕ := [4, 9]

-- Define the movement probabilities
def move_prob : ℚ := 1/3
def skip_one_prob : ℚ := 1/3
def skip_two_prob : ℚ := 1/3

-- Function to calculate the probability of reaching the target pad
def reach_target_prob : ℚ := sorry

-- Theorem stating the probability of reaching the target pad
theorem reach_target_probability :
  reach_target_prob = 1/729 :=
sorry

end NUMINAMATH_CALUDE_reach_target_probability_l1281_128116


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1281_128147

def U : Set ℤ := {-1, 1, 2, 3, 4}
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {2, 4}

theorem complement_union_theorem :
  (Set.compl A ∩ U) ∪ B = {-1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1281_128147


namespace NUMINAMATH_CALUDE_g_function_equality_l1281_128134

theorem g_function_equality (x : ℝ) :
  let g : ℝ → ℝ := λ x => -4*x^5 + 4*x^3 - 4*x + 6
  4*x^5 + 3*x^3 + x - 2 + g x = 7*x^3 - 5*x + 4 :=
by
  sorry

end NUMINAMATH_CALUDE_g_function_equality_l1281_128134


namespace NUMINAMATH_CALUDE_work_completion_time_l1281_128154

/-- Given workers A, B, and C with their individual work rates, 
    prove that B and C together can complete the work in 3 hours. -/
theorem work_completion_time 
  (rate_A : ℝ) 
  (rate_B : ℝ) 
  (rate_C : ℝ) 
  (h1 : rate_A = 1 / 4) 
  (h2 : rate_A + rate_C = 1 / 2) 
  (h3 : rate_B = 1 / 12) : 
  1 / (rate_B + rate_C) = 3 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1281_128154


namespace NUMINAMATH_CALUDE_junior_score_theorem_l1281_128141

theorem junior_score_theorem (n : ℝ) (h : n > 0) :
  let junior_ratio : ℝ := 0.2
  let senior_ratio : ℝ := 0.8
  let total_average : ℝ := 80
  let senior_average : ℝ := 78
  let junior_count : ℝ := junior_ratio * n
  let senior_count : ℝ := senior_ratio * n
  let total_score : ℝ := total_average * n
  let senior_total_score : ℝ := senior_average * senior_count
  let junior_total_score : ℝ := total_score - senior_total_score
  junior_total_score / junior_count = 88 :=
by sorry

end NUMINAMATH_CALUDE_junior_score_theorem_l1281_128141


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1281_128109

theorem fraction_subtraction : 
  (5 + 7 + 9) / (2 + 4 + 6) - (4 + 6 + 8) / (3 + 5 + 7) = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1281_128109


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1281_128170

/-- The perimeter of a regular hexagon with side length 5 meters is 30 meters. -/
theorem regular_hexagon_perimeter : 
  ∀ (side_length : ℝ), 
  side_length = 5 → 
  (6 : ℝ) * side_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1281_128170


namespace NUMINAMATH_CALUDE_last_digit_power_difference_l1281_128130

def last_digit (n : ℤ) : ℕ := (n % 10).toNat

theorem last_digit_power_difference (x : ℤ) :
  last_digit (x^95 - 3^58) = 4 → last_digit (x^95) = 3 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_power_difference_l1281_128130


namespace NUMINAMATH_CALUDE_third_place_winnings_value_l1281_128115

/-- The amount of money in the pot -/
def pot_total : ℝ := 210

/-- The percentage of the pot that the third place winner receives -/
def third_place_percentage : ℝ := 0.15

/-- The amount of money the third place winner receives -/
def third_place_winnings : ℝ := pot_total * third_place_percentage

theorem third_place_winnings_value : third_place_winnings = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_third_place_winnings_value_l1281_128115


namespace NUMINAMATH_CALUDE_vet_donation_l1281_128118

theorem vet_donation (dog_fee : ℕ) (cat_fee : ℕ) (dog_adoptions : ℕ) (cat_adoptions : ℕ) 
  (h1 : dog_fee = 15)
  (h2 : cat_fee = 13)
  (h3 : dog_adoptions = 8)
  (h4 : cat_adoptions = 3) :
  (dog_fee * dog_adoptions + cat_fee * cat_adoptions) / 3 = 53 := by
  sorry

end NUMINAMATH_CALUDE_vet_donation_l1281_128118


namespace NUMINAMATH_CALUDE_inequality_a_l1281_128193

theorem inequality_a (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 * y^2 * z + y^4 * x^2 * z + y^4 * z^2 * x + z^4 * y^2 * x + x^4 * z^2 * y + z^4 * x^2 * y ≥
  2 * (x^3 * y^2 * z^2 + x^2 * y^3 * z^2 + x^2 * y^2 * z^3) := by
sorry


end NUMINAMATH_CALUDE_inequality_a_l1281_128193


namespace NUMINAMATH_CALUDE_triangle_min_angle_le_60_l1281_128163

theorem triangle_min_angle_le_60 (A B C : ℝ) : 
  A + B + C = 180 → A > 0 → B > 0 → C > 0 → min A (min B C) ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_min_angle_le_60_l1281_128163


namespace NUMINAMATH_CALUDE_simplify_fraction_l1281_128199

theorem simplify_fraction (b : ℚ) (h : b = 2) : (15 * b^4) / (75 * b^3) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1281_128199


namespace NUMINAMATH_CALUDE_mailing_weight_calculation_l1281_128121

/-- The total weight of a mailing with multiple envelopes and additional materials -/
def total_mailing_weight (envelope_weight : ℝ) (num_envelopes : ℕ) (additional_weight : ℝ) : ℝ :=
  (envelope_weight + additional_weight) * num_envelopes

/-- Theorem stating that the total weight of the mailing is 9240 grams -/
theorem mailing_weight_calculation :
  total_mailing_weight 8.5 880 2 = 9240 := by
  sorry

end NUMINAMATH_CALUDE_mailing_weight_calculation_l1281_128121


namespace NUMINAMATH_CALUDE_abs_eq_self_not_negative_l1281_128187

theorem abs_eq_self_not_negative (x : ℝ) : |x| = x → x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_self_not_negative_l1281_128187


namespace NUMINAMATH_CALUDE_rectangular_hall_length_l1281_128119

/-- 
Proves that for a rectangular hall where the breadth is two-thirds of the length 
and the area is 2400 sq metres, the length is 60 metres.
-/
theorem rectangular_hall_length 
  (length breadth : ℝ) 
  (breadth_relation : breadth = (2/3) * length)
  (area : ℝ)
  (area_calculation : area = length * breadth)
  (given_area : area = 2400) : 
  length = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_length_l1281_128119


namespace NUMINAMATH_CALUDE_domain_relationship_l1281_128177

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2x+1)
def domain_f_2x_plus_1 : Set ℝ := Set.Icc (-3) 3

-- Theorem stating the relationship between the domains
theorem domain_relationship :
  (∀ y ∈ domain_f_2x_plus_1, ∃ x, y = 2*x + 1) →
  {x : ℝ | f x ≠ 0} = Set.Icc (-5) 7 :=
sorry

end NUMINAMATH_CALUDE_domain_relationship_l1281_128177


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l1281_128120

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/15 = 1

-- Define the left focus of the hyperbola
def left_focus : ℝ × ℝ := (-4, 0)

-- Define the line that the circle is tangent to
def tangent_line (x : ℝ) : Prop := x = 4

-- Define the moving circle
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_focus : center.1 - 4 = center.2
  tangent_to_line : center.1 + center.2 = 4

-- Theorem statement
theorem trajectory_is_parabola (M : MovingCircle) :
  (M.center.2)^2 = -16 * M.center.1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l1281_128120


namespace NUMINAMATH_CALUDE_carl_driving_hours_l1281_128180

/-- Calculates the total driving hours for Carl over two weeks -/
def total_driving_hours : ℕ :=
  let daily_hours : ℕ := 2
  let days_in_two_weeks : ℕ := 14
  let additional_weekly_hours : ℕ := 6
  let weeks : ℕ := 2
  (daily_hours * days_in_two_weeks) + (additional_weekly_hours * weeks)

/-- Theorem stating that Carl's total driving hours over two weeks is 40 -/
theorem carl_driving_hours : total_driving_hours = 40 := by
  sorry

end NUMINAMATH_CALUDE_carl_driving_hours_l1281_128180


namespace NUMINAMATH_CALUDE_x_minus_p_equals_five_minus_two_p_l1281_128101

theorem x_minus_p_equals_five_minus_two_p (x p : ℝ) 
  (h1 : |x - 5| = p) (h2 : x < 5) : x - p = 5 - 2*p := by
  sorry

end NUMINAMATH_CALUDE_x_minus_p_equals_five_minus_two_p_l1281_128101


namespace NUMINAMATH_CALUDE_train_length_l1281_128117

/-- Proves that a train with the given conditions has a length of 1500 meters -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) : 
  train_speed = 180 * (1000 / 3600) →  -- Convert 180 km/hr to m/s
  crossing_time = 60 →  -- Convert 1 minute to seconds
  train_length * 2 = train_speed * crossing_time →
  train_length = 1500 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1281_128117


namespace NUMINAMATH_CALUDE_triangle_area_after_10_seconds_l1281_128190

/-- Represents the position of a runner at time t -/
def RunnerPosition (t : ℝ) := ℝ × ℝ

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Represents the position of a runner over time -/
structure Runner where
  initialPos : ℝ × ℝ
  velocity : ℝ

/-- Calculates the position of a runner at time t -/
def runnerPosition (r : Runner) (t : ℝ) : RunnerPosition t := sorry

theorem triangle_area_after_10_seconds
  (a b c : Runner)
  (h1 : triangleArea (runnerPosition a 0) (runnerPosition b 0) (runnerPosition c 0) = 2)
  (h2 : triangleArea (runnerPosition a 5) (runnerPosition b 5) (runnerPosition c 5) = 3) :
  (triangleArea (runnerPosition a 10) (runnerPosition b 10) (runnerPosition c 10) = 4) ∨
  (triangleArea (runnerPosition a 10) (runnerPosition b 10) (runnerPosition c 10) = 8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_after_10_seconds_l1281_128190


namespace NUMINAMATH_CALUDE_dave_chocolate_boxes_l1281_128178

theorem dave_chocolate_boxes (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) : 
  total_boxes = 12 → pieces_per_box = 3 → pieces_left = 21 →
  (total_boxes * pieces_per_box - pieces_left) / pieces_per_box = 5 := by
sorry

end NUMINAMATH_CALUDE_dave_chocolate_boxes_l1281_128178


namespace NUMINAMATH_CALUDE_common_chord_length_l1281_128194

/-- Curve C1 defined by (x-1)^2 + y^2 = 4 -/
def C1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

/-- Curve C2 defined by x^2 + y^2 = 4y -/
def C2 (x y : ℝ) : Prop := x^2 + y^2 = 4*y

/-- The length of the common chord between C1 and C2 is √11 -/
theorem common_chord_length :
  ∃ (a b c d : ℝ), C1 a b ∧ C1 c d ∧ C2 a b ∧ C2 c d ∧
  ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = Real.sqrt 11 :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l1281_128194


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_relation_l1281_128151

-- Define a structure for a trapezoid
structure Trapezoid where
  a : ℝ  -- larger base
  c : ℝ  -- smaller base
  e : ℝ  -- diagonal
  f : ℝ  -- diagonal
  d : ℝ  -- side
  b : ℝ  -- side
  h_ac : a > c  -- condition that a > c

-- Theorem statement
theorem trapezoid_diagonal_relation (T : Trapezoid) :
  (T.e^2 + T.f^2) / (T.a^2 - T.b^2) = (T.a + T.c) / (T.a - T.c) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_relation_l1281_128151


namespace NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l1281_128106

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fourth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a2 : a 2 = 4)
  (h_a3 : a 3 = 6) :
  a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l1281_128106


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l1281_128122

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

-- Define the solution set of the quadratic inequality
def solution_set (a : ℕ → ℝ) : Set ℝ :=
  {x : ℝ | x^2 - a 3 * x + a 4 ≤ 0}

-- Theorem statement
theorem arithmetic_sequence_theorem (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  solution_set a = {x : ℝ | a 1 ≤ x ∧ x ≤ a 2} →
  ∀ n : ℕ, a n = 2 * n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l1281_128122


namespace NUMINAMATH_CALUDE_gcd_459_357_l1281_128131

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l1281_128131


namespace NUMINAMATH_CALUDE_overhead_percentage_problem_l1281_128114

/-- Calculates the percentage of cost for overhead given the purchase price, markup, and net profit. -/
def overhead_percentage (purchase_price markup net_profit : ℚ) : ℚ :=
  let overhead := markup - net_profit
  (overhead / purchase_price) * 100

/-- Theorem stating that given the specific values in the problem, the overhead percentage is 37.5% -/
theorem overhead_percentage_problem :
  let purchase_price : ℚ := 48
  let markup : ℚ := 30
  let net_profit : ℚ := 12
  overhead_percentage purchase_price markup net_profit = 37.5 := by
  sorry

#eval overhead_percentage 48 30 12

end NUMINAMATH_CALUDE_overhead_percentage_problem_l1281_128114


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l1281_128162

-- Define the points
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (7, 0)
def D : ℝ × ℝ := (0, 4)

-- Define the length of AC
def AC : ℝ := 15

-- Theorem statement
theorem area_of_triangle_ABC :
  let A : ℝ × ℝ := (0, 4) -- We know A is on the y-axis at (0,4) because D is at (0,4)
  (1/2 : ℝ) * ‖(A.1 - B.1, A.2 - B.2)‖ * ‖(C.1 - B.1, C.2 - B.2)‖ = 2 * Real.sqrt 209 := by
sorry


end NUMINAMATH_CALUDE_area_of_triangle_ABC_l1281_128162


namespace NUMINAMATH_CALUDE_unique_two_digit_number_with_reverse_difference_64_l1281_128161

theorem unique_two_digit_number_with_reverse_difference_64 :
  ∃! N : ℕ, 
    (N ≥ 10 ∧ N < 100) ∧ 
    (∃ a : ℕ, a < 10 ∧ N = 10 * a + 1) ∧
    ((10 * (N % 10) + N / 10) - N = 64) := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_with_reverse_difference_64_l1281_128161


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l1281_128158

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = 5 * x - 20
def line2 (x y : ℝ) : Prop := 3 * x + y = 110

-- Define the intersection point
def intersection (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ x y : ℝ, intersection x y ∧ x = 16.25 := by
sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l1281_128158


namespace NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l1281_128129

-- Define a geometric sequence
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

-- Theorem statement
theorem eighth_term_of_geometric_sequence :
  let a := 4
  let a2 := 16
  let r := a2 / a
  geometric_sequence a r 8 = 65536 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l1281_128129


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_range_of_a_l1281_128169

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem 1
theorem intersection_A_B : A ∩ B = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2
theorem union_A_complement_B : A ∪ (U \ B) = {x | x < 4} := by sorry

-- Theorem 3
theorem range_of_a (h : A ⊆ C a) : a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_range_of_a_l1281_128169


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1281_128172

theorem sqrt_equation_solution :
  ∀ x : ℚ, (x > 2) → (Real.sqrt (8 * x) / Real.sqrt (5 * (x - 2)) = 3) → x = 90 / 37 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1281_128172


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1281_128140

theorem polar_to_rectangular_conversion :
  let r : ℝ := 3 * Real.sqrt 2
  let θ : ℝ := 5 * Real.pi / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -3 * Real.sqrt 6 / 2) ∧ (y = 3 * Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1281_128140


namespace NUMINAMATH_CALUDE_inequality_of_positive_reals_l1281_128197

theorem inequality_of_positive_reals (a b c d e f : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f) :
  (a * b) / (a + b) + (c * d) / (c + d) + (e * f) / (e + f) ≤ 
  ((a + c + e) * (b + d + f)) / (a + b + c + d + e + f) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_reals_l1281_128197
