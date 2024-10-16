import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_2870_l1542_154248

theorem sum_of_squares_2870 :
  ∃! (n : ℕ), n > 0 ∧ n * (n + 1) * (2 * n + 1) / 6 = 2870 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_2870_l1542_154248


namespace NUMINAMATH_CALUDE_circle_construction_l1542_154271

/-- Given four lines intersecting at a point with 45° angles between them, and a circle
    intersecting these lines such that two opposite chords have lengths a and b,
    and one chord is three times the length of its opposite chord,
    the circle's center (u, v) and radius r satisfy specific equations. -/
theorem circle_construction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (u v r : ℝ),
    u^2 = (a^2 - b^2) / 8 + Real.sqrt (((a^2 - b^2) / 8)^2 + ((a^2 + b^2) / 10)^2) ∧
    v^2 = r^2 - a^2 / 4 ∧
    r^2 = (u^2 + v^2) / 2 + (a^2 + b^2) / 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_construction_l1542_154271


namespace NUMINAMATH_CALUDE_unique_quadratic_root_l1542_154224

theorem unique_quadratic_root (m : ℝ) : 
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) → m = 0 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_root_l1542_154224


namespace NUMINAMATH_CALUDE_log_6_15_in_terms_of_a_b_l1542_154296

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the logarithm with arbitrary base
noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem statement
theorem log_6_15_in_terms_of_a_b (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) :
  log 6 15 = (b + 1 - a) / (a + b) := by
  sorry


end NUMINAMATH_CALUDE_log_6_15_in_terms_of_a_b_l1542_154296


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l1542_154268

/-- An arithmetic sequence -/
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic a)
  (h_sum : a 2 + a 8 = 16)
  (h_fourth : a 4 = 6) :
  a 6 = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l1542_154268


namespace NUMINAMATH_CALUDE_circle_equation_l1542_154205

/-- A circle C with center (0, a) -/
structure Circle (a : ℝ) where
  center : ℝ × ℝ := (0, a)

/-- The equation of a circle with center (0, a) and radius r -/
def circleEquation (c : Circle a) (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + (y - c.center.2)^2 = r^2

/-- The circle passes through the point (1, 0) -/
def passesThrough (c : Circle a) (r : ℝ) : Prop :=
  circleEquation c r 1 0

/-- The circle is divided by the x-axis into two arcs with length ratio 1:2 -/
def arcRatio (c : Circle a) : Prop :=
  abs (a / 1) = Real.sqrt 3

theorem circle_equation (a : ℝ) (c : Circle a) (h1 : passesThrough c (Real.sqrt (4/3)))
    (h2 : arcRatio c) :
    ∀ x y : ℝ, circleEquation c (Real.sqrt (4/3)) x y ↔ 
      x^2 + (y - Real.sqrt 3 / 3)^2 = 4/3 ∨ x^2 + (y + Real.sqrt 3 / 3)^2 = 4/3 :=
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1542_154205


namespace NUMINAMATH_CALUDE_rugby_team_average_weight_l1542_154259

theorem rugby_team_average_weight 
  (initial_players : ℕ) 
  (new_player_weight : ℝ) 
  (new_average_weight : ℝ) : 
  initial_players = 20 ∧ 
  new_player_weight = 210 ∧ 
  new_average_weight = 181.42857142857142 → 
  (initial_players * (new_average_weight * (initial_players + 1) - new_player_weight)) / initial_players = 180 := by
sorry

end NUMINAMATH_CALUDE_rugby_team_average_weight_l1542_154259


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1542_154200

theorem divisibility_theorem (a b c d m : ℤ) 
  (h_odd : Odd m)
  (h_div_sum : m ∣ (a + b + c + d))
  (h_div_sum_squares : m ∣ (a^2 + b^2 + c^2 + d^2)) :
  m ∣ (a^4 + b^4 + c^4 + d^4 + 4*a*b*c*d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1542_154200


namespace NUMINAMATH_CALUDE_regression_line_equation_l1542_154234

/-- Given a regression line with slope 1.23 passing through the point (4, 5),
    prove that its equation is ŷ = 1.23x + 0.08 -/
theorem regression_line_equation (slope : ℝ) (center_x center_y : ℝ) :
  slope = 1.23 →
  center_x = 4 →
  center_y = 5 →
  ∃ (intercept : ℝ), 
    intercept = center_y - slope * center_x ∧
    intercept = 0.08 ∧
    ∀ (x y : ℝ), y = slope * x + intercept := by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l1542_154234


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l1542_154237

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (1 + Complex.I) * (1 + a * Complex.I)
  (z.re = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l1542_154237


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1542_154292

theorem sum_of_solutions_is_zero (x : ℝ) :
  ((-12 * x) / (x^2 - 1) = (3 * x) / (x + 1) - 9 / (x - 1)) →
  (∃ y : ℝ, (-12 * y) / (y^2 - 1) = (3 * y) / (y + 1) - 9 / (y - 1) ∧ y ≠ x) →
  x + y = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1542_154292


namespace NUMINAMATH_CALUDE_square_one_implies_plus_minus_one_l1542_154221

theorem square_one_implies_plus_minus_one (x : ℝ) : x^2 = 1 → x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_one_implies_plus_minus_one_l1542_154221


namespace NUMINAMATH_CALUDE_figure_area_theorem_l1542_154263

theorem figure_area_theorem (x : ℝ) :
  let square1_area := (3 * x)^2
  let square2_area := (7 * x)^2
  let triangle_area := (1 / 2) * (3 * x) * (7 * x)
  square1_area + square2_area + triangle_area = 1300 →
  x = Real.sqrt (2600 / 137) := by
sorry

end NUMINAMATH_CALUDE_figure_area_theorem_l1542_154263


namespace NUMINAMATH_CALUDE_linear_system_solution_existence_l1542_154272

theorem linear_system_solution_existence
  (a b c d : ℤ)
  (h_nonzero : a * d - b * c ≠ 0)
  (b₁ b₂ : ℤ)
  (h_b₁ : ∃ k : ℤ, b₁ = (a * d - b * c) * k)
  (h_b₂ : ∃ q : ℤ, b₂ = (a * d - b * c) * q) :
  ∃ x y : ℤ, a * x + b * y = b₁ ∧ c * x + d * y = b₂ :=
sorry

end NUMINAMATH_CALUDE_linear_system_solution_existence_l1542_154272


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1542_154294

/-- The number of different arrangements for selecting 4 students (1 girl and 3 boys) 
    from 8 students (6 boys and 2 girls) by stratified sampling based on gender, 
    with a girl as the first runner. -/
def stratifiedSamplingArrangements : ℕ := sorry

/-- The total number of students -/
def totalStudents : ℕ := 8

/-- The number of boys -/
def numBoys : ℕ := 6

/-- The number of girls -/
def numGirls : ℕ := 2

/-- The number of students to be selected -/
def selectedStudents : ℕ := 4

/-- The number of boys to be selected -/
def selectedBoys : ℕ := 3

/-- The number of girls to be selected -/
def selectedGirls : ℕ := 1

theorem stratified_sampling_theorem : 
  stratifiedSamplingArrangements = 240 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1542_154294


namespace NUMINAMATH_CALUDE_jace_driving_problem_l1542_154236

/-- Jace's driving problem -/
theorem jace_driving_problem (speed : ℝ) (break_time : ℝ) (second_drive_time : ℝ) (total_distance : ℝ) :
  speed = 60 ∧ 
  break_time = 0.5 ∧ 
  second_drive_time = 9 ∧ 
  total_distance = 780 →
  ∃ (first_drive_time : ℝ), 
    first_drive_time * speed + second_drive_time * speed = total_distance ∧ 
    first_drive_time = 4 :=
by sorry

end NUMINAMATH_CALUDE_jace_driving_problem_l1542_154236


namespace NUMINAMATH_CALUDE_problem_statement_l1542_154290

theorem problem_statement (x y z : ℝ) 
  (h1 : x ≤ y) (h2 : y ≤ z) 
  (h3 : x + y + z = 12) 
  (h4 : x^2 + y^2 + z^2 = 54) : 
  x ≤ 3 ∧ z ≥ 5 ∧ 
  9 ≤ x * y ∧ x * y ≤ 25 ∧ 
  9 ≤ y * z ∧ y * z ≤ 25 ∧ 
  9 ≤ z * x ∧ z * x ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1542_154290


namespace NUMINAMATH_CALUDE_stock_yield_proof_l1542_154222

/-- Proves that the calculated yield matches the quoted yield for a given stock --/
theorem stock_yield_proof (quoted_yield : ℚ) (stock_price : ℚ) 
  (h1 : quoted_yield = 8 / 100)
  (h2 : stock_price = 225) : 
  let dividend := quoted_yield * stock_price
  ((dividend / stock_price) * 100 : ℚ) = quoted_yield * 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_yield_proof_l1542_154222


namespace NUMINAMATH_CALUDE_divisibility_of_square_sum_minus_2017_l1542_154284

theorem divisibility_of_square_sum_minus_2017 (n : ℕ) : 
  ∃ x y : ℤ, (n : ℤ) ∣ (x^2 + y^2 - 2017) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_square_sum_minus_2017_l1542_154284


namespace NUMINAMATH_CALUDE_percent_of_12356_l1542_154254

theorem percent_of_12356 (p : ℝ) : p * 12356 = 1.2356 → p * 100 = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_12356_l1542_154254


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l1542_154261

theorem simplify_trigonometric_expression (α : Real) 
  (h1 : π < α ∧ α < 3*π/2) : 
  Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) - 
  Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) = 
  -2 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l1542_154261


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_l1542_154262

/-- If the terminal side of angle α passes through point (-2, 1), then 1/(sin 2α) = -5/4 -/
theorem angle_terminal_side_point (α : ℝ) : 
  (Real.cos α = -2 / Real.sqrt 5 ∧ Real.sin α = 1 / Real.sqrt 5) → 
  1 / Real.sin (2 * α) = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_l1542_154262


namespace NUMINAMATH_CALUDE_minimize_reciprocal_sum_l1542_154253

theorem minimize_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 30) :
  (1 / a + 4 / b ≥ 8 / 15) ∧
  (1 / a + 4 / b = 8 / 15 ↔ a = 15 / 4 ∧ b = 15) := by
sorry

end NUMINAMATH_CALUDE_minimize_reciprocal_sum_l1542_154253


namespace NUMINAMATH_CALUDE_dessert_coffee_probability_l1542_154260

theorem dessert_coffee_probability
  (p_dessert_and_coffee : ℝ)
  (p_no_dessert : ℝ)
  (h1 : p_dessert_and_coffee = 0.6)
  (h2 : p_no_dessert = 0.2500000000000001) :
  p_dessert_and_coffee + (1 - p_no_dessert - p_dessert_and_coffee) = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_dessert_coffee_probability_l1542_154260


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l1542_154246

theorem floor_ceiling_sum : ⌊(-3.72 : ℝ)⌋ + ⌈(34.1 : ℝ)⌉ = 31 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l1542_154246


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l1542_154252

theorem hemisphere_surface_area (r : ℝ) (h : r = 6) :
  let sphere_area := λ r : ℝ => 4 * π * r^2
  let base_area := π * r^2
  let hemisphere_area := sphere_area r / 2 + base_area
  hemisphere_area = 108 * π := by sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l1542_154252


namespace NUMINAMATH_CALUDE_missing_number_is_34_l1542_154243

theorem missing_number_is_34 : 
  ∃ x : ℝ, ((306 / x) * 15 + 270 = 405) ∧ (x = 34) :=
by sorry

end NUMINAMATH_CALUDE_missing_number_is_34_l1542_154243


namespace NUMINAMATH_CALUDE_fraction_equality_l1542_154230

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 2 / 3) 
  (h2 : c / b = 1 / 5) 
  (h3 : c / d = 7 / 15) : 
  a * b / (c * d) = 140 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1542_154230


namespace NUMINAMATH_CALUDE_eggs_per_hen_l1542_154219

/-- Given 303.0 eggs collected from 28.0 hens, prove that the number of eggs
    laid by each hen, when rounded to the nearest whole number, is 11. -/
theorem eggs_per_hen (total_eggs : ℝ) (num_hens : ℝ) 
    (h1 : total_eggs = 303.0) (h2 : num_hens = 28.0) :
  round (total_eggs / num_hens) = 11 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_hen_l1542_154219


namespace NUMINAMATH_CALUDE_stratified_sampling_middle_managers_l1542_154255

theorem stratified_sampling_middle_managers :
  ∀ (total_employees : ℕ) 
    (middle_managers : ℕ) 
    (sample_size : ℕ),
  total_employees = 160 →
  middle_managers = 30 →
  sample_size = 32 →
  (middle_managers * sample_size) / total_employees = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_middle_managers_l1542_154255


namespace NUMINAMATH_CALUDE_sqrt_negative_eight_a_cubed_l1542_154276

theorem sqrt_negative_eight_a_cubed (a : ℝ) (h : a ≤ 0) :
  Real.sqrt (-8 * a^3) = -2 * a * Real.sqrt (-2 * a) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_negative_eight_a_cubed_l1542_154276


namespace NUMINAMATH_CALUDE_like_terms_sum_l1542_154206

theorem like_terms_sum (a b : ℝ) (x y : ℝ) 
  (h : 3 * a^(7*x) * b^(y+7) = 5 * a^(2-4*y) * b^(2*x)) : x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_l1542_154206


namespace NUMINAMATH_CALUDE_a_formula_S_min_l1542_154282

-- Define the sequence and its sum
def S (n : ℕ) : ℤ := n^2 - 48*n

def a : ℕ → ℤ
  | 0 => 0  -- We define a₀ = 0 to make a total function
  | n + 1 => S (n + 1) - S n

-- Theorem for the general formula of a_n
theorem a_formula (n : ℕ) : a (n + 1) = 2 * (n + 1) - 49 := by sorry

-- Theorem for the minimum value of S_n
theorem S_min : ∃ n : ℕ, S n = -576 ∧ ∀ m : ℕ, S m ≥ -576 := by sorry

end NUMINAMATH_CALUDE_a_formula_S_min_l1542_154282


namespace NUMINAMATH_CALUDE_not_always_same_direction_for_parallel_vectors_l1542_154204

-- Define a vector type
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define parallel vectors
def parallel (u v : V) : Prop :=
  ∃ k : ℝ, v = k • u

-- Theorem statement
theorem not_always_same_direction_for_parallel_vectors :
  ¬ ∀ (u v : V), parallel u v → (∃ k : ℝ, k > 0 ∧ v = k • u) :=
sorry

end NUMINAMATH_CALUDE_not_always_same_direction_for_parallel_vectors_l1542_154204


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1542_154251

-- Define a structure for a rectangular solid
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ

-- Define properties of the rectangular solid
def isPrime (n : ℕ) : Prop := sorry

def volume (r : RectangularSolid) : ℕ :=
  r.length * r.width * r.height

def surfaceArea (r : RectangularSolid) : ℕ :=
  2 * (r.length * r.width + r.width * r.height + r.height * r.length)

-- Theorem statement
theorem rectangular_solid_surface_area :
  ∀ (r : RectangularSolid),
    isPrime r.length ∧ isPrime r.width ∧ isPrime r.height →
    volume r = 1155 →
    surfaceArea r = 142 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1542_154251


namespace NUMINAMATH_CALUDE_donalds_oranges_l1542_154211

theorem donalds_oranges (initial : ℕ) : initial + 5 = 9 → initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_donalds_oranges_l1542_154211


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1542_154220

/-- Given two vectors a and b in ℝ², if a is parallel to b and a = (1, -2) and b = (x, 1), then x = -1/2 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (1, -2))
  (h2 : b = (x, 1))
  (h_parallel : ∃ (k : ℝ), a = k • b) :
  x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1542_154220


namespace NUMINAMATH_CALUDE_percentage_problem_l1542_154258

theorem percentage_problem (x : ℝ) (P : ℝ) : 
  x = 150 → 
  P * x = 0.20 * 487.50 → 
  P = 0.65 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1542_154258


namespace NUMINAMATH_CALUDE_no_nonzero_ending_product_zero_l1542_154293

theorem no_nonzero_ending_product_zero (x y : ℤ) : 
  (x % 10 ≠ 0) → (y % 10 ≠ 0) → (x * y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_nonzero_ending_product_zero_l1542_154293


namespace NUMINAMATH_CALUDE_range_of_m_l1542_154226

theorem range_of_m (x y m : ℝ) : 
  (2 * x - y = 5 * m) →
  (3 * x + 4 * y = 2 * m) →
  (x + y ≤ 5) →
  (2 * x + 7 * y < 18) →
  (-6 < m ∧ m ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1542_154226


namespace NUMINAMATH_CALUDE_smallest_group_size_exists_group_size_l1542_154285

theorem smallest_group_size (n : ℕ) : 
  (n % 6 = 1) ∧ (n % 9 = 3) ∧ (n % 8 = 5) → n ≥ 169 :=
by sorry

theorem exists_group_size : 
  ∃ n : ℕ, (n % 6 = 1) ∧ (n % 9 = 3) ∧ (n % 8 = 5) ∧ n = 169 :=
by sorry

end NUMINAMATH_CALUDE_smallest_group_size_exists_group_size_l1542_154285


namespace NUMINAMATH_CALUDE_otimes_nested_equal_101_l1542_154233

-- Define the operation ⊗
def otimes (a b : ℚ) : ℚ := b^2 + 1

-- Theorem statement
theorem otimes_nested_equal_101 (m : ℚ) : otimes m (otimes m 3) = 101 := by
  sorry

end NUMINAMATH_CALUDE_otimes_nested_equal_101_l1542_154233


namespace NUMINAMATH_CALUDE_opposite_roots_quadratic_l1542_154207

theorem opposite_roots_quadratic (k : ℝ) : 
  (∃ x y : ℝ, x^2 + (k^2 - 4)*x + k - 1 = 0 ∧ 
               y^2 + (k^2 - 4)*y + k - 1 = 0 ∧ 
               x = -y) → 
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_opposite_roots_quadratic_l1542_154207


namespace NUMINAMATH_CALUDE_irrational_arithmetic_properties_l1542_154257

-- Define irrational numbers
def IsIrrational (x : ℝ) : Prop := ¬ (∃ (q : ℚ), (x : ℝ) = q)

-- Theorem statement
theorem irrational_arithmetic_properties :
  (∃ (a b : ℝ), IsIrrational a ∧ IsIrrational b ∧ IsIrrational (a + b)) ∧
  (∃ (a b : ℝ), IsIrrational a ∧ IsIrrational b ∧ ∃ (q : ℚ), (a - b : ℝ) = q) ∧
  (∃ (a b : ℝ), IsIrrational a ∧ IsIrrational b ∧ ∃ (q : ℚ), (a * b : ℝ) = q) ∧
  (∃ (a b : ℝ), IsIrrational a ∧ IsIrrational b ∧ b ≠ 0 ∧ ∃ (q : ℚ), (a / b : ℝ) = q) :=
by sorry

end NUMINAMATH_CALUDE_irrational_arithmetic_properties_l1542_154257


namespace NUMINAMATH_CALUDE_rectangular_plot_ratio_l1542_154209

theorem rectangular_plot_ratio (length breadth area : ℝ) : 
  breadth = 14 →
  area = 588 →
  area = length * breadth →
  length / breadth = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_ratio_l1542_154209


namespace NUMINAMATH_CALUDE_golden_fish_catches_l1542_154241

theorem golden_fish_catches (x y z : ℕ) : 
  4 * x + 2 * z = 1000 →
  2 * y + z = 800 →
  x + y + z = 900 :=
by sorry

end NUMINAMATH_CALUDE_golden_fish_catches_l1542_154241


namespace NUMINAMATH_CALUDE_x_varies_as_z_two_thirds_l1542_154265

/-- Given that x varies directly as the square of y, and y varies directly as the cube root of z,
    prove that x varies as z^(2/3). -/
theorem x_varies_as_z_two_thirds
  (x y z : ℝ)
  (h1 : ∃ k : ℝ, ∀ y, x = k * y^2)
  (h2 : ∃ j : ℝ, ∀ z, y = j * z^(1/3))
  : ∃ m : ℝ, x = m * z^(2/3) :=
sorry

end NUMINAMATH_CALUDE_x_varies_as_z_two_thirds_l1542_154265


namespace NUMINAMATH_CALUDE_modified_lucas_60th_term_mod_5_l1542_154297

def modifiedLucas : ℕ → ℤ
  | 0 => 2
  | 1 => 5
  | n + 2 => modifiedLucas n + modifiedLucas (n + 1)

theorem modified_lucas_60th_term_mod_5 :
  modifiedLucas 59 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modified_lucas_60th_term_mod_5_l1542_154297


namespace NUMINAMATH_CALUDE_cos_equality_with_period_l1542_154228

theorem cos_equality_with_period (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → 
  Real.cos (n * π / 180) = Real.cos (845 * π / 180) → 
  n = 125 := by
sorry

end NUMINAMATH_CALUDE_cos_equality_with_period_l1542_154228


namespace NUMINAMATH_CALUDE_cube_volume_l1542_154232

theorem cube_volume (edge_sum : ℝ) (h : edge_sum = 96) : 
  let edge_length := edge_sum / 12
  let volume := edge_length ^ 3
  volume = 512 := by sorry

end NUMINAMATH_CALUDE_cube_volume_l1542_154232


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1542_154213

theorem stratified_sampling_theorem (grade10 : ℕ) (grade11 : ℕ) (grade12 : ℕ) 
  (sample12 : ℕ) (h1 : grade10 = 1600) (h2 : grade11 = 1200) (h3 : grade12 = 800) 
  (h4 : sample12 = 20) :
  let total_upper := grade10 + grade11
  let total_sample_upper := (total_upper * sample12) / grade12
  total_sample_upper = 70 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1542_154213


namespace NUMINAMATH_CALUDE_day_after_53_days_from_friday_l1542_154212

/-- Days of the week represented as an enumeration --/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week --/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after a given number of days --/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

/-- Theorem: The day of the week 53 days after Friday is Tuesday --/
theorem day_after_53_days_from_friday :
  dayAfter DayOfWeek.Friday 53 = DayOfWeek.Tuesday := by
  sorry


end NUMINAMATH_CALUDE_day_after_53_days_from_friday_l1542_154212


namespace NUMINAMATH_CALUDE_two_pairs_probability_l1542_154239

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 6

/-- The probability of rolling exactly two pairs of dice showing the same value,
    with the other two dice each showing different numbers that don't match the paired numbers,
    when rolling six standard six-sided dice once -/
def probabilityTwoPairs : ℚ :=
  25 / 72

theorem two_pairs_probability :
  probabilityTwoPairs = (
    (numFaces.choose 2) *
    (numDice.choose 2) *
    ((numDice - 2).choose 2) *
    (numFaces - 2) *
    (numFaces - 3)
  ) / (numFaces ^ numDice) :=
sorry

end NUMINAMATH_CALUDE_two_pairs_probability_l1542_154239


namespace NUMINAMATH_CALUDE_diamond_eight_five_l1542_154280

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := (a + b) * ((a - b)^2)

-- Theorem statement
theorem diamond_eight_five : diamond 8 5 = 117 := by
  sorry

end NUMINAMATH_CALUDE_diamond_eight_five_l1542_154280


namespace NUMINAMATH_CALUDE_triangular_number_200_l1542_154229

/-- Triangular number sequence -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 200th triangular number is 20100 -/
theorem triangular_number_200 : triangular_number 200 = 20100 := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_200_l1542_154229


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l1542_154286

-- Define a point in the grid
structure Point where
  x : Nat
  y : Nat

-- Define the grid size
def gridSize : Nat := 6

-- Define the initially shaded squares
def initialShaded : List Point := [
  { x := 1, y := 1 },
  { x := 1, y := 6 },
  { x := 6, y := 1 },
  { x := 3, y := 4 }
]

-- Function to check if a point is within the grid
def inGrid (p : Point) : Bool :=
  1 ≤ p.x ∧ p.x ≤ gridSize ∧ 1 ≤ p.y ∧ p.y ≤ gridSize

-- Function to check if a set of points has both horizontal and vertical symmetry
def hasSymmetry (points : List Point) : Bool :=
  sorry

-- The main theorem
theorem min_additional_squares_for_symmetry :
  ∃ (additionalPoints : List Point),
    additionalPoints.length = 4 ∧
    (∀ p ∈ additionalPoints, inGrid p) ∧
    hasSymmetry (initialShaded ++ additionalPoints) ∧
    (∀ (otherPoints : List Point),
      otherPoints.length < 4 →
      ¬ hasSymmetry (initialShaded ++ otherPoints)) :=
  sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l1542_154286


namespace NUMINAMATH_CALUDE_square_sum_given_condition_l1542_154269

theorem square_sum_given_condition (a b : ℝ) : 
  a^2 * b^2 + a^2 + b^2 + 16 = 10 * a * b → a^2 + b^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_condition_l1542_154269


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l1542_154299

/-- Geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_prop : a 2 * a 4 = a 5)
  (h_a4 : a 4 = 8) :
  ∃ (q : ℝ) (S : ℕ → ℝ),
    (q = 2) ∧
    (∀ n : ℕ, S n = 2^n - 1) ∧
    (∀ n : ℕ, S n = (a 1) * (1 - q^n) / (1 - q)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l1542_154299


namespace NUMINAMATH_CALUDE_log_function_value_l1542_154235

-- Define the logarithmic function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_function_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (f a (1/8) = 3) → (f a (1/4) = 2) :=
by sorry

end NUMINAMATH_CALUDE_log_function_value_l1542_154235


namespace NUMINAMATH_CALUDE_polynomial_equality_l1542_154215

theorem polynomial_equality : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1542_154215


namespace NUMINAMATH_CALUDE_expression_evaluation_l1542_154216

theorem expression_evaluation :
  let a : ℚ := -1/3
  let b : ℚ := -3
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1542_154216


namespace NUMINAMATH_CALUDE_negation_equivalence_l1542_154244

theorem negation_equivalence (m : ℝ) :
  (¬ ∃ (x : ℤ), x^2 + x + m < 0) ↔ (∀ (x : ℝ), x^2 + x + m ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1542_154244


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1542_154208

theorem fraction_equals_zero (x : ℝ) : (x - 1) / (3 * x + 1) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1542_154208


namespace NUMINAMATH_CALUDE_farmer_profit_l1542_154270

/-- Calculate the profit for a group of piglets -/
def profit_for_group (num_piglets : ℕ) (months : ℕ) (price : ℕ) : ℕ :=
  num_piglets * price - num_piglets * 12 * months

/-- Calculate the total profit for all piglet groups -/
def total_profit : ℕ :=
  profit_for_group 2 12 350 +
  profit_for_group 3 15 400 +
  profit_for_group 2 18 450 +
  profit_for_group 1 21 500

/-- The farmer's profit from selling 8 piglets is $1788 -/
theorem farmer_profit : total_profit = 1788 := by
  sorry

end NUMINAMATH_CALUDE_farmer_profit_l1542_154270


namespace NUMINAMATH_CALUDE_function_lower_bound_l1542_154250

/-- Given a function f(x) = (1/2)x^4 - 2x^3 + 3m for all real x,
    if f(x) + 9 ≥ 0 for all real x, then m ≥ 3/2 --/
theorem function_lower_bound (m : ℝ) : 
  (∀ x : ℝ, (1/2) * x^4 - 2 * x^3 + 3 * m + 9 ≥ 0) → m ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l1542_154250


namespace NUMINAMATH_CALUDE_sqrt_product_l1542_154225

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by sorry

end NUMINAMATH_CALUDE_sqrt_product_l1542_154225


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1542_154256

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 3) + 1 / (b + 3) = 1 / 4) : 
  a + 3 * b ≥ 4 + 8 * Real.sqrt 3 ∧ 
  (a + 3 * b = 4 + 8 * Real.sqrt 3 ↔ a = 1 + 4 * Real.sqrt 3 ∧ b = (3 + 4 * Real.sqrt 3) / 3) := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1542_154256


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_in_closed_interval_l1542_154289

/-- A function f : ℝ → ℝ is increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The cosine function -/
noncomputable def cos : ℝ → ℝ := Real.cos

/-- The function f(x) = x - a * cos(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * cos x

theorem increasing_f_implies_a_in_closed_interval :
  ∀ a : ℝ, IsIncreasing (f a) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_in_closed_interval_l1542_154289


namespace NUMINAMATH_CALUDE_omitted_angle_measure_l1542_154287

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180° --/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- The property that the sum of interior angles is divisible by 180° --/
def is_valid_sum (s : ℕ) : Prop := ∃ k : ℕ, s = k * 180

/-- The sum calculated by Angela --/
def angela_sum : ℕ := 2583

/-- The theorem to prove --/
theorem omitted_angle_measure :
  ∃ (n : ℕ), 
    n > 2 ∧ 
    is_valid_sum (sum_interior_angles n) ∧ 
    sum_interior_angles n = angela_sum + 117 := by
  sorry

end NUMINAMATH_CALUDE_omitted_angle_measure_l1542_154287


namespace NUMINAMATH_CALUDE_rational_trig_sums_l1542_154201

theorem rational_trig_sums (x : ℝ) 
  (s_rational : ∃ q : ℚ, (Real.sin (64 * x) + Real.sin (65 * x)) = ↑q)
  (t_rational : ∃ q : ℚ, (Real.cos (64 * x) + Real.cos (65 * x)) = ↑q) :
  (∃ q1 q2 : ℚ, Real.cos (64 * x) = ↑q1 ∧ Real.cos (65 * x) = ↑q2) ∨
  (∃ q1 q2 : ℚ, Real.sin (64 * x) = ↑q1 ∧ Real.sin (65 * x) = ↑q2) :=
by sorry

end NUMINAMATH_CALUDE_rational_trig_sums_l1542_154201


namespace NUMINAMATH_CALUDE_time_on_other_subjects_is_40_l1542_154249

/-- Represents the time spent on homework for each subject -/
structure HomeworkTime where
  total : ℝ
  math : ℝ
  science : ℝ
  history : ℝ
  english : ℝ

/-- Calculates the time spent on other subjects -/
def timeOnOtherSubjects (hw : HomeworkTime) : ℝ :=
  hw.total - (hw.math + hw.science + hw.history + hw.english)

/-- Theorem stating the time spent on other subjects is 40 minutes -/
theorem time_on_other_subjects_is_40 (hw : HomeworkTime) : 
  hw.total = 150 ∧
  hw.math = 0.20 * hw.total ∧
  hw.science = 0.25 * hw.total ∧
  hw.history = 0.10 * hw.total ∧
  hw.english = 0.15 * hw.total ∧
  hw.history ≥ 20 ∧
  hw.science ≥ 20 →
  timeOnOtherSubjects hw = 40 := by
  sorry

#check time_on_other_subjects_is_40

end NUMINAMATH_CALUDE_time_on_other_subjects_is_40_l1542_154249


namespace NUMINAMATH_CALUDE_toy_ratio_l1542_154214

def total_toys : ℕ := 240
def elder_son_toys : ℕ := 60

def younger_son_toys : ℕ := total_toys - elder_son_toys

theorem toy_ratio :
  (younger_son_toys : ℚ) / elder_son_toys = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_toy_ratio_l1542_154214


namespace NUMINAMATH_CALUDE_f_range_implies_m_plus_n_range_l1542_154274

-- Define the function f(x)
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x

-- Define the interval [m, n]
def interval (m n : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ n }

-- State the theorem
theorem f_range_implies_m_plus_n_range (m n : ℝ) :
  (∀ x ∈ interval m n, -6 ≤ f x ∧ f x ≤ 2) →
  (0 ≤ m + n ∧ m + n ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_f_range_implies_m_plus_n_range_l1542_154274


namespace NUMINAMATH_CALUDE_smallest_seven_digit_divisible_by_127_l1542_154281

theorem smallest_seven_digit_divisible_by_127 : 
  ∀ n : ℕ, n ≥ 1000000 → n.mod 127 = 0 → n ≥ 1000125 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_seven_digit_divisible_by_127_l1542_154281


namespace NUMINAMATH_CALUDE_area_closer_to_vertex_is_one_third_l1542_154267

/-- An equilateral triangle -/
structure EquilateralTriangle where
  -- We don't need to define the specifics of the triangle,
  -- just that it exists and is equilateral

/-- The centroid of a triangle -/
def centroid (t : EquilateralTriangle) : Point := sorry

/-- A point in the plane of the triangle -/
structure PointInTriangle (t : EquilateralTriangle) where
  point : Point

/-- The area of the equilateral triangle -/
def area (t : EquilateralTriangle) : ℝ := sorry

/-- The area of the region where points are closer to the nearest vertex than to the centroid -/
def area_closer_to_vertex (t : EquilateralTriangle) : ℝ := sorry

/-- The theorem stating that the area where points are closer to the nearest vertex
    is 1/3 of the total area of the equilateral triangle -/
theorem area_closer_to_vertex_is_one_third (t : EquilateralTriangle) :
  area_closer_to_vertex t = (1/3) * area t := by sorry

end NUMINAMATH_CALUDE_area_closer_to_vertex_is_one_third_l1542_154267


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l1542_154247

-- Define the sum of the first n even numbers
def sumEven (n : ℕ) : ℕ := n * (n + 1)

-- Define the sum of the first n odd numbers
def sumOdd (n : ℕ) : ℕ := n^2

-- State the theorem
theorem even_odd_sum_difference : sumEven 100 - sumOdd 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l1542_154247


namespace NUMINAMATH_CALUDE_system_solution_l1542_154275

theorem system_solution (x y : ℝ) (eq1 : 2*x + y = 5) (eq2 : x + 2*y = 4) : x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1542_154275


namespace NUMINAMATH_CALUDE_preceding_binary_number_l1542_154217

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNumber := List Nat

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + bit * 2^i) 0

/-- Converts a decimal number to its binary representation -/
def decimal_to_binary (n : Nat) : BinaryNumber :=
  sorry

theorem preceding_binary_number (M : BinaryNumber) :
  M = [0, 1, 0, 1, 0, 0, 1] →
  decimal_to_binary (binary_to_decimal M - 1) = [1, 0, 0, 1, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_preceding_binary_number_l1542_154217


namespace NUMINAMATH_CALUDE_hundredth_term_equals_981_l1542_154298

/-- Sequence of powers of 3 or sums of distinct powers of 3 -/
def PowerOf3Sequence : ℕ → ℕ :=
  sorry

/-- The 100th term of the PowerOf3Sequence -/
def HundredthTerm : ℕ := PowerOf3Sequence 100

theorem hundredth_term_equals_981 : HundredthTerm = 981 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_equals_981_l1542_154298


namespace NUMINAMATH_CALUDE_crayons_left_l1542_154283

/-- Represents the number of crayons Mary has -/
structure Crayons where
  green : Nat
  blue : Nat

/-- Calculates the total number of crayons -/
def total_crayons (c : Crayons) : Nat :=
  c.green + c.blue

/-- Represents the number of crayons Mary gives away -/
structure CrayonsGiven where
  green : Nat
  blue : Nat

/-- Calculates the total number of crayons given away -/
def total_given (g : CrayonsGiven) : Nat :=
  g.green + g.blue

/-- Theorem: Mary has 9 crayons left after giving some away -/
theorem crayons_left (initial : Crayons) (given : CrayonsGiven) 
  (h1 : initial.green = 5)
  (h2 : initial.blue = 8)
  (h3 : given.green = 3)
  (h4 : given.blue = 1) :
  total_crayons initial - total_given given = 9 := by
  sorry


end NUMINAMATH_CALUDE_crayons_left_l1542_154283


namespace NUMINAMATH_CALUDE_abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two_l1542_154218

theorem abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two :
  (∀ x : ℝ, x < -2 → |x| > 2) ∧
  (∃ x : ℝ, |x| > 2 ∧ ¬(x < -2)) :=
sorry

end NUMINAMATH_CALUDE_abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two_l1542_154218


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_necessary_not_sufficient_condition_l1542_154273

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - 2*a*x + 2*a^2 - a - 6

-- Define the proposition p
def has_real_roots (a : ℝ) : Prop := ∃ x : ℝ, quadratic a x = 0

-- Define the proposition q
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 3

theorem quadratic_roots_condition (a : ℝ) :
  ¬(has_real_roots a) ↔ (a < -2 ∨ a > 3) := by sorry

theorem necessary_not_sufficient_condition (m : ℝ) :
  (∀ a : ℝ, q m a → has_real_roots a) ∧
  (∃ a : ℝ, has_real_roots a ∧ ¬(q m a)) →
  -1 ≤ m ∧ m < 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_necessary_not_sufficient_condition_l1542_154273


namespace NUMINAMATH_CALUDE_negation_of_existence_l1542_154279

variable (a : ℝ)

theorem negation_of_existence (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a*x + 1 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_existence_l1542_154279


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l1542_154278

/-- Calculates the total amount collected from ticket sales given the number of adults and children, and their respective ticket prices. -/
def totalTicketSales (numAdults numChildren adultPrice childPrice : ℕ) : ℕ :=
  numAdults * adultPrice + numChildren * childPrice

/-- Theorem stating that given the specific conditions of the problem, the total ticket sales amount to $246. -/
theorem theater_ticket_sales :
  let adultPrice : ℕ := 11
  let childPrice : ℕ := 10
  let totalAttendees : ℕ := 23
  let numChildren : ℕ := 7
  let numAdults : ℕ := totalAttendees - numChildren
  totalTicketSales numAdults numChildren adultPrice childPrice = 246 :=
by
  sorry

#check theater_ticket_sales

end NUMINAMATH_CALUDE_theater_ticket_sales_l1542_154278


namespace NUMINAMATH_CALUDE_cube_face_sum_l1542_154210

/-- Represents the numbers on the faces of a cube -/
structure CubeNumbers where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  e : ℕ+
  f : ℕ+
  g : ℕ+
  h : ℕ+
  vertex_sum_eq : (a + e) * (b + f) * (c + g) * h = 2002

/-- The sum of numbers on the faces of the cube is 39 -/
theorem cube_face_sum (cube : CubeNumbers) : 
  cube.a + cube.b + cube.c + cube.e + cube.f + cube.g + cube.h = 39 := by
  sorry


end NUMINAMATH_CALUDE_cube_face_sum_l1542_154210


namespace NUMINAMATH_CALUDE_f_value_at_2013_l1542_154291

theorem f_value_at_2013 (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = a * x^3 + b * Real.sin x + 9) →
  f (-2013) = 7 →
  f 2013 = 11 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_2013_l1542_154291


namespace NUMINAMATH_CALUDE_solution_sets_equality_l1542_154202

theorem solution_sets_equality (a b : ℝ) : 
  (∀ x, |x - 2| > 1 ↔ x^2 + a*x + b > 0) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_equality_l1542_154202


namespace NUMINAMATH_CALUDE_initial_tickets_count_l1542_154240

/-- The number of tickets sold in the first week -/
def first_week_sales : ℕ := 38

/-- The number of tickets sold in the second week -/
def second_week_sales : ℕ := 17

/-- The number of tickets left to sell -/
def remaining_tickets : ℕ := 35

/-- The initial number of tickets -/
def initial_tickets : ℕ := first_week_sales + second_week_sales + remaining_tickets

theorem initial_tickets_count : initial_tickets = 90 := by
  sorry

end NUMINAMATH_CALUDE_initial_tickets_count_l1542_154240


namespace NUMINAMATH_CALUDE_function_machine_output_l1542_154295

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 25 then
    step1 - 7
  else
    (step1 + 3) * 2

theorem function_machine_output : function_machine 12 = 78 := by
  sorry

end NUMINAMATH_CALUDE_function_machine_output_l1542_154295


namespace NUMINAMATH_CALUDE_rectangle_area_l1542_154277

theorem rectangle_area (w : ℝ) (h₁ : w > 0) : 
  let l := 4 * w
  let perimeter := 2 * l + 2 * w
  perimeter = 200 → l * w = 1600 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1542_154277


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1542_154266

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality : 
  (¬ ∃ x : ℝ, x^2 + 4*x + 6 < 0) ↔ (∀ x : ℝ, x^2 + 4*x + 6 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1542_154266


namespace NUMINAMATH_CALUDE_joe_money_left_l1542_154288

/-- The amount of money Joe has left after shopping and donating to charity -/
def money_left (initial_amount notebooks books pens stickers notebook_price book_price pen_price sticker_price charity : ℕ) : ℕ :=
  initial_amount - (notebooks * notebook_price + books * book_price + pens * pen_price + stickers * sticker_price + charity)

/-- Theorem stating that Joe has $60 left after his shopping trip and charity donation -/
theorem joe_money_left :
  money_left 150 7 2 5 3 4 12 2 6 10 = 60 := by
  sorry

#eval money_left 150 7 2 5 3 4 12 2 6 10

end NUMINAMATH_CALUDE_joe_money_left_l1542_154288


namespace NUMINAMATH_CALUDE_incorrect_calculation_l1542_154223

theorem incorrect_calculation (a : ℝ) : (2 * a)^3 ≠ 6 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l1542_154223


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1542_154238

/-- Given a hyperbola with equation y^2 - x^2/4 = 1, its asymptotes have the equation y = ± x/2 -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (y^2 - x^2/4 = 1) → (∃ (k : ℝ), k = x/2 ∧ (y = k ∨ y = -k)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1542_154238


namespace NUMINAMATH_CALUDE_unique_natural_pair_l1542_154203

theorem unique_natural_pair : 
  ∃! (k n : ℕ), 
    120 < k * n ∧ k * n < 130 ∧ 
    2 < (k : ℚ) / n ∧ (k : ℚ) / n < 3 ∧
    k = 18 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_natural_pair_l1542_154203


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l1542_154231

/-- Pete's current age -/
def p : ℕ := sorry

/-- Mandy's current age -/
def m : ℕ := sorry

/-- The number of years until the ratio of their ages is 3:2 -/
def x : ℕ := sorry

/-- Pete's age two years ago was twice Mandy's age two years ago -/
axiom past_condition_1 : p - 2 = 2 * (m - 2)

/-- Pete's age four years ago was three times Mandy's age four years ago -/
axiom past_condition_2 : p - 4 = 3 * (m - 4)

/-- The ratio of their ages will be 3:2 after x years -/
axiom future_ratio : (p + x) / (m + x) = 3 / 2

theorem age_ratio_in_two_years :
  x = 2 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l1542_154231


namespace NUMINAMATH_CALUDE_adrian_water_needed_l1542_154242

/-- Represents the recipe ratios and amount of orange juice used --/
structure Recipe where
  water_sugar_ratio : ℚ
  sugar_juice_ratio : ℚ
  orange_juice_cups : ℚ

/-- Calculates the amount of water needed for the punch recipe --/
def water_needed (r : Recipe) : ℚ :=
  r.water_sugar_ratio * r.sugar_juice_ratio * r.orange_juice_cups

/-- Theorem stating that Adrian needs 60 cups of water --/
theorem adrian_water_needed :
  let recipe := Recipe.mk 5 3 4
  water_needed recipe = 60 := by
  sorry

end NUMINAMATH_CALUDE_adrian_water_needed_l1542_154242


namespace NUMINAMATH_CALUDE_simplify_expression_l1542_154245

theorem simplify_expression (a b : ℝ) (h : a + b ≠ 0) :
  a - b + (2 * b^2) / (a + b) = (a^2 + b^2) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1542_154245


namespace NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l1542_154264

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + Real.sqrt 2

-- Define the condition for intersection points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧ 
    hyperbola_C A.1 A.2 ∧ hyperbola_C B.1 B.2 ∧
    line_l k A.1 A.2 ∧ line_l k B.1 B.2

-- Define the dot product condition
def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 > 2

-- Main theorem
theorem hyperbola_intersection_theorem (k : ℝ) :
  (hyperbola_C 0 0) ∧  -- Center at origin
  (hyperbola_C 2 0) ∧  -- Right focus at (2,0)
  (hyperbola_C (Real.sqrt 3) 0) ∧  -- Right vertex at (√3,0)
  (intersects_at_two_points k) ∧
  (∀ A B : ℝ × ℝ, hyperbola_C A.1 A.2 ∧ hyperbola_C B.1 B.2 ∧
    line_l k A.1 A.2 ∧ line_l k B.1 B.2 → dot_product_condition A B) →
  (-1 < k ∧ k < -Real.sqrt 3 / 3) ∨ (Real.sqrt 3 / 3 < k ∧ k < 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l1542_154264


namespace NUMINAMATH_CALUDE_square_sum_range_l1542_154227

theorem square_sum_range (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 1) :
  1/2 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_range_l1542_154227
