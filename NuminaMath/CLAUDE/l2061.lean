import Mathlib

namespace NUMINAMATH_CALUDE_lisa_caffeine_limit_l2061_206164

/-- The amount of caffeine (in mg) in one cup of coffee -/
def caffeine_per_cup : ℕ := 80

/-- The number of cups of coffee Lisa drinks -/
def cups_of_coffee : ℕ := 3

/-- The amount (in mg) by which Lisa exceeds her daily limit when drinking three cups -/
def excess_caffeine : ℕ := 40

/-- Lisa's daily caffeine limit (in mg) -/
def lisas_caffeine_limit : ℕ := 200

theorem lisa_caffeine_limit :
  lisas_caffeine_limit = cups_of_coffee * caffeine_per_cup - excess_caffeine :=
by sorry

end NUMINAMATH_CALUDE_lisa_caffeine_limit_l2061_206164


namespace NUMINAMATH_CALUDE_k_set_characterization_l2061_206105

/-- Given h = 2^r for some non-negative integer r, k(h) is the set of all natural numbers k
    such that there exist an odd natural number m > 1 and a natural number n where
    k divides m^k - 1 and m divides n^((n^k - 1)/k) + 1 -/
def k_set (h : ℕ) : Set ℕ :=
  {k : ℕ | ∃ (m n : ℕ), m > 1 ∧ m % 2 = 1 ∧ 
    (m^k - 1) % k = 0 ∧ (n^((n^k - 1)/k) + 1) % m = 0}

/-- For h = 2^r, the set k(h) is equal to {2^(r+s) * t | s, t ∈ ℕ, t is odd} -/
theorem k_set_characterization (r : ℕ) :
  k_set (2^r) = {k : ℕ | ∃ (s t : ℕ), k = 2^(r+s) * t ∧ t % 2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_k_set_characterization_l2061_206105


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2061_206122

theorem triangle_angle_calculation (a b c : Real) (A B C : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  -- Law of sines: a / sin(A) = b / sin(B) = c / sin(C)
  (a / Real.sin A = b / Real.sin B) →
  -- Given conditions
  (a = 3) →
  (b = Real.sqrt 6) →
  (A = 2 * Real.pi / 3) →
  -- Conclusion
  B = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2061_206122


namespace NUMINAMATH_CALUDE_weight_after_deliveries_l2061_206160

/-- Calculates the remaining weight on a truck after two deliveries with given percentages -/
def remaining_weight (initial_load : ℝ) (first_unload_percent : ℝ) (second_unload_percent : ℝ) : ℝ :=
  let remaining_after_first := initial_load * (1 - first_unload_percent)
  remaining_after_first * (1 - second_unload_percent)

/-- Theorem stating the remaining weight after two deliveries -/
theorem weight_after_deliveries :
  remaining_weight 50000 0.1 0.2 = 36000 := by
  sorry

end NUMINAMATH_CALUDE_weight_after_deliveries_l2061_206160


namespace NUMINAMATH_CALUDE_complex_number_operation_l2061_206124

theorem complex_number_operation : 
  let z₁ : ℂ := -2 + 5*I
  let z₂ : ℂ := 3*I
  3 * z₁ + z₂ = -6 + 18*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_operation_l2061_206124


namespace NUMINAMATH_CALUDE_walking_problem_l2061_206120

/-- Proves that given the conditions of the walking problem, the speed of the second man is 3 km/h -/
theorem walking_problem (distance : ℝ) (speed_first : ℝ) (time_diff : ℝ) :
  distance = 6 →
  speed_first = 4 →
  time_diff = 0.5 →
  let time_first := distance / speed_first
  let time_second := time_first + time_diff
  let speed_second := distance / time_second
  speed_second = 3 := by
  sorry

end NUMINAMATH_CALUDE_walking_problem_l2061_206120


namespace NUMINAMATH_CALUDE_equality_of_expressions_l2061_206106

theorem equality_of_expressions : 
  (-2^2 ≠ (-2)^2) ∧ 
  (2^3 ≠ 3^2) ∧ 
  (-3^3 = (-3)^3) ∧ 
  (-|-3| ≠ -(-3)) := by
  sorry

end NUMINAMATH_CALUDE_equality_of_expressions_l2061_206106


namespace NUMINAMATH_CALUDE_tan_ratio_sum_l2061_206179

theorem tan_ratio_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_sum_l2061_206179


namespace NUMINAMATH_CALUDE_a_value_l2061_206101

theorem a_value (a : ℝ) : 2 ∈ ({1, a, a^2 - a} : Set ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l2061_206101


namespace NUMINAMATH_CALUDE_tiles_per_row_l2061_206119

/-- Given a square room with an area of 256 square feet and tiles that are 8 inches wide,
    prove that 24 tiles can fit in one row. -/
theorem tiles_per_row (room_area : ℝ) (tile_width : ℝ) : 
  room_area = 256 → tile_width = 8 / 12 → (Real.sqrt room_area) * 12 / tile_width = 24 := by
  sorry

end NUMINAMATH_CALUDE_tiles_per_row_l2061_206119


namespace NUMINAMATH_CALUDE_triangle_angle_A_l2061_206136

theorem triangle_angle_A (A : Real) (h : 4 * Real.pi * Real.sin A - 3 * Real.arccos (-1/2) = 0) :
  A = Real.pi / 6 ∨ A = 5 * Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l2061_206136


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2061_206154

theorem complex_fraction_equality : Complex.I ^ 2 = -1 → (1 + Complex.I) ^ 2 / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2061_206154


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2061_206197

theorem trigonometric_identity (α : Real) 
  (h1 : α > 0) 
  (h2 : α < π / 3) 
  (h3 : Real.sqrt 3 * Real.sin α + Real.cos α = Real.sqrt 6 / 2) : 
  (Real.cos (α + π / 6) = Real.sqrt 10 / 4) ∧ 
  (Real.cos (2 * α + 7 * π / 12) = (Real.sqrt 2 - Real.sqrt 30) / 8) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2061_206197


namespace NUMINAMATH_CALUDE_difference_max_min_both_languages_l2061_206123

/-- The number of students studying both Spanish and French -/
def students_both (S F : ℕ) : ℤ := S + F - 2001

theorem difference_max_min_both_languages :
  ∃ (S_min S_max F_min F_max : ℕ),
    1601 ≤ S_min ∧ S_max ≤ 1700 ∧
    601 ≤ F_min ∧ F_max ≤ 800 ∧
    (∀ S F, 1601 ≤ S ∧ S ≤ 1700 ∧ 601 ≤ F ∧ F ≤ 800 →
      students_both S_min F_min ≤ students_both S F ∧
      students_both S F ≤ students_both S_max F_max) ∧
    students_both S_max F_max - students_both S_min F_min = 298 := by
  sorry

end NUMINAMATH_CALUDE_difference_max_min_both_languages_l2061_206123


namespace NUMINAMATH_CALUDE_floating_time_calculation_l2061_206134

/-- Floating time calculation -/
theorem floating_time_calculation
  (boat_speed_with_current : ℝ)
  (boat_speed_against_current : ℝ)
  (distance_floated : ℝ)
  (h1 : boat_speed_with_current = 28)
  (h2 : boat_speed_against_current = 24)
  (h3 : distance_floated = 20) :
  (distance_floated / ((boat_speed_with_current - boat_speed_against_current) / 2)) = 10 := by
  sorry

#check floating_time_calculation

end NUMINAMATH_CALUDE_floating_time_calculation_l2061_206134


namespace NUMINAMATH_CALUDE_f_properties_l2061_206141

-- Define the function f(x) = -x|x| + 2x
def f (x : ℝ) : ℝ := -x * abs x + 2 * x

-- State the theorem
theorem f_properties :
  -- f is an odd function
  (∀ x, f (-x) = -f x) ∧
  -- f is monotonically decreasing on (-∞, -1)
  (∀ x y, x < y → y < -1 → f y < f x) ∧
  -- f is monotonically decreasing on (1, +∞)
  (∀ x y, 1 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2061_206141


namespace NUMINAMATH_CALUDE_sandy_final_position_l2061_206167

-- Define the coordinate system
def Position := ℝ × ℝ

-- Define the starting position
def start : Position := (0, 0)

-- Define Sandy's movements
def move_south (p : Position) (distance : ℝ) : Position :=
  (p.1, p.2 - distance)

def move_east (p : Position) (distance : ℝ) : Position :=
  (p.1 + distance, p.2)

def move_north (p : Position) (distance : ℝ) : Position :=
  (p.1, p.2 + distance)

-- Define Sandy's final position after her movements
def final_position : Position :=
  move_east (move_north (move_east (move_south start 20) 20) 20) 20

-- Theorem to prove
theorem sandy_final_position :
  final_position = (40, 0) := by sorry

end NUMINAMATH_CALUDE_sandy_final_position_l2061_206167


namespace NUMINAMATH_CALUDE_sum_of_multiples_l2061_206198

def largest_two_digit_multiple_of_5 : ℕ :=
  95

def smallest_three_digit_multiple_of_7 : ℕ :=
  105

theorem sum_of_multiples : 
  largest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l2061_206198


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2061_206142

/-- Given a geometric sequence with positive terms and common ratio 2, 
    if the product of the 3rd and 11th terms is 16, then the 10th term is 32. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = 2 * a n) →
  a 3 * a 11 = 16 →
  a 10 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2061_206142


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2061_206138

theorem line_segment_endpoint (y : ℝ) : 
  y > 0 → 
  Real.sqrt ((7 - 2)^2 + (y - 4)^2) = 6 → 
  y = 4 + Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2061_206138


namespace NUMINAMATH_CALUDE_min_value_theorem_l2061_206107

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 4 / b) ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1 / a₀ + 4 / b₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2061_206107


namespace NUMINAMATH_CALUDE_range_of_a_l2061_206135

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 4) → (x^2 - 2*x + 1 - a^2 < 0)) →
  (a > 0) →
  (a ≥ 3) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2061_206135


namespace NUMINAMATH_CALUDE_similarity_transformation_result_l2061_206165

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the similarity transformation
def similarityTransform (p : Point2D) (ratio : ℝ) : Point2D :=
  { x := p.x * ratio, y := p.y * ratio }

-- Define the theorem
theorem similarity_transformation_result :
  let A : Point2D := { x := -4, y := 2 }
  let B : Point2D := { x := -6, y := -4 }
  let ratio : ℝ := 1/2
  let A' := similarityTransform A ratio
  (A'.x = -2 ∧ A'.y = 1) ∨ (A'.x = 2 ∧ A'.y = -1) :=
sorry

end NUMINAMATH_CALUDE_similarity_transformation_result_l2061_206165


namespace NUMINAMATH_CALUDE_sugar_for_partial_recipe_result_as_mixed_number_l2061_206151

-- Define the original amount of sugar in the recipe
def original_sugar : ℚ := 19/3

-- Define the fraction of the recipe we want to make
def recipe_fraction : ℚ := 1/3

-- Theorem statement
theorem sugar_for_partial_recipe :
  recipe_fraction * original_sugar = 19/9 :=
by sorry

-- Convert the result to a mixed number
theorem result_as_mixed_number :
  ∃ (whole : ℕ) (num denom : ℕ), 
    recipe_fraction * original_sugar = whole + num / denom ∧
    whole = 2 ∧ num = 1 ∧ denom = 9 :=
by sorry

end NUMINAMATH_CALUDE_sugar_for_partial_recipe_result_as_mixed_number_l2061_206151


namespace NUMINAMATH_CALUDE_rectangle_folding_l2061_206137

theorem rectangle_folding (a b : ℝ) : 
  a = 5 ∧ 
  0 < b ∧ 
  b < 4 ∧ 
  (a - b)^2 + b^2 = 6 → 
  b = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_folding_l2061_206137


namespace NUMINAMATH_CALUDE_num_beakers_is_three_l2061_206196

def volume_per_tube : ℚ := 7
def num_tubes : ℕ := 6
def volume_per_beaker : ℚ := 14

theorem num_beakers_is_three : 
  (volume_per_tube * num_tubes) / volume_per_beaker = 3 := by
  sorry

end NUMINAMATH_CALUDE_num_beakers_is_three_l2061_206196


namespace NUMINAMATH_CALUDE_inequality_proof_l2061_206189

theorem inequality_proof (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 6) 
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 12) : 
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧ 
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2061_206189


namespace NUMINAMATH_CALUDE_rectangle_area_l2061_206187

/-- Theorem: For a rectangle EFGH with vertices E(0, 0), F(0, y), G(y, 3y), and H(y, 0),
    where y > 0 and the area of the rectangle is 45 square units, the value of y is √15. -/
theorem rectangle_area (y : ℝ) (h1 : y > 0) (h2 : y * 3 * y = 45) : y = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2061_206187


namespace NUMINAMATH_CALUDE_smallest_x_cosine_equality_l2061_206112

theorem smallest_x_cosine_equality : ∃ x : ℝ, 
  (x > 0 ∧ x < 30) ∧ 
  (∀ y : ℝ, y > 0 ∧ y < 30 ∧ Real.cos (3 * y * π / 180) = Real.cos ((2 * y^2 - y) * π / 180) → x ≤ y) ∧
  Real.cos (3 * x * π / 180) = Real.cos ((2 * x^2 - x) * π / 180) ∧
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_cosine_equality_l2061_206112


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_ratio_quadratic_function_minimum_ratio_exact_l2061_206109

/-- A quadratic function f(x) = ax^2 + bx + c satisfying certain conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  second_derivative_positive : 2 * a > 0
  non_negative : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0

/-- The theorem stating the minimum value of f(1) / f''(0) for the quadratic function -/
theorem quadratic_function_minimum_ratio (f : QuadraticFunction) :
  (f.a + f.b + f.c) / (2 * f.a) ≥ 2 := by
  sorry

/-- The theorem stating that the minimum value of f(1) / f''(0) is exactly 2 -/
theorem quadratic_function_minimum_ratio_exact :
  ∃ f : QuadraticFunction, (f.a + f.b + f.c) / (2 * f.a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_ratio_quadratic_function_minimum_ratio_exact_l2061_206109


namespace NUMINAMATH_CALUDE_rectangle_diagonal_length_l2061_206195

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  area : ℝ
  diagonal : ℝ

-- State the theorem
theorem rectangle_diagonal_length 
  (rect : Rectangle) 
  (h1 : rect.length = 16) 
  (h2 : rect.area = 192) 
  (h3 : rect.area = rect.length * rect.width) 
  (h4 : rect.diagonal ^ 2 = rect.length ^ 2 + rect.width ^ 2) : 
  rect.diagonal = 20 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_diagonal_length_l2061_206195


namespace NUMINAMATH_CALUDE_parallelogram_to_rhombus_transformation_l2061_206169

def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (-3, -1)
def D : ℝ × ℝ := (1, -1)

def M (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![k, 1; 0, 2]

def is_parallelogram (A B C D : ℝ × ℝ) : Prop :=
  (A.1 - B.1 = D.1 - C.1) ∧ (A.2 - B.2 = D.2 - C.2) ∧
  (A.1 - D.1 = B.1 - C.1) ∧ (A.2 - D.2 = B.2 - C.2)

def is_rhombus (A B C D : ℝ × ℝ) : Prop :=
  let AB := ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let DA := ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  AB = BC ∧ BC = CD ∧ CD = DA

def transform_point (M : Matrix (Fin 2) (Fin 2) ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (M 0 0 * p.1 + M 0 1 * p.2, M 1 0 * p.1 + M 1 1 * p.2)

theorem parallelogram_to_rhombus_transformation (k : ℝ) :
  k < 0 →
  is_parallelogram A B C D →
  is_rhombus (transform_point (M k) A) (transform_point (M k) B)
             (transform_point (M k) C) (transform_point (M k) D) →
  k = -1 ∧ M k⁻¹ = !![(-1 : ℝ), (1/2 : ℝ); 0, (1/2 : ℝ)] :=
sorry

end NUMINAMATH_CALUDE_parallelogram_to_rhombus_transformation_l2061_206169


namespace NUMINAMATH_CALUDE_groups_created_is_four_l2061_206194

/-- The number of groups created when dividing insects equally -/
def number_of_groups (boys_insects : ℕ) (girls_insects : ℕ) (insects_per_group : ℕ) : ℕ :=
  (boys_insects + girls_insects) / insects_per_group

/-- Theorem stating that the number of groups is 4 given the specific conditions -/
theorem groups_created_is_four :
  number_of_groups 200 300 125 = 4 := by
  sorry

end NUMINAMATH_CALUDE_groups_created_is_four_l2061_206194


namespace NUMINAMATH_CALUDE_triangle_area_l2061_206180

theorem triangle_area (A B C : ℝ) (b : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  b = 2 ∧  -- Given condition
  B = π / 3 ∧  -- Given condition
  Real.sin (2 * A) + Real.sin (A - C) - Real.sin B = 0 →  -- Given equation
  (1 / 2) * b * b * Real.sin B = Real.sqrt 3  -- Area of the triangle
  := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2061_206180


namespace NUMINAMATH_CALUDE_line_slope_l2061_206118

/-- Given a line y = kx + 1 passing through points (4, b), (a, 5), and (a, b + 1),
    where a and b are real numbers, the value of k is 3/4. -/
theorem line_slope (k a b : ℝ) : 
  (b = 4 * k + 1) →
  (5 = a * k + 1) →
  (b + 1 = a * k + 1) →
  k = 3/4 := by sorry

end NUMINAMATH_CALUDE_line_slope_l2061_206118


namespace NUMINAMATH_CALUDE_constant_term_in_system_l2061_206117

theorem constant_term_in_system (x y C : ℝ) : 
  (5 * x + y = 19) →
  (x + 3 * y = C) →
  (3 * x + 2 * y = 10) →
  C = 1 := by
sorry

end NUMINAMATH_CALUDE_constant_term_in_system_l2061_206117


namespace NUMINAMATH_CALUDE_fifth_term_value_l2061_206178

def sequence_term (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => 2^i + 5)

theorem fifth_term_value : sequence_term 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l2061_206178


namespace NUMINAMATH_CALUDE_tan_addition_formula_l2061_206104

theorem tan_addition_formula (x : Real) (h : Real.tan x = 3) :
  Real.tan (x + π / 6) = 5 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_addition_formula_l2061_206104


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l2061_206188

/-- The number of sides in a heptagon -/
def heptagon_sides : ℕ := 7

/-- Formula for the number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex heptagon is 14 -/
theorem heptagon_diagonals : num_diagonals heptagon_sides = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l2061_206188


namespace NUMINAMATH_CALUDE_hyperbola_right_angle_area_l2061_206132

/-- The hyperbola with equation x²/9 - y²/16 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := (-5, 0)

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := (5, 0)

/-- A point on the hyperbola -/
def P : ℝ × ℝ := sorry

/-- The angle ∠F₁PF₂ is 90° -/
def right_angle : Prop :=
  let v₁ := (P.1 - F₁.1, P.2 - F₁.2)
  let v₂ := (P.1 - F₂.1, P.2 - F₂.2)
  v₁.1 * v₂.1 + v₁.2 * v₂.2 = 0

/-- The area of triangle ΔF₁PF₂ -/
def triangle_area : ℝ := sorry

theorem hyperbola_right_angle_area :
  hyperbola P.1 P.2 →
  right_angle →
  triangle_area = 16 := by sorry

end NUMINAMATH_CALUDE_hyperbola_right_angle_area_l2061_206132


namespace NUMINAMATH_CALUDE_intersection_distance_l2061_206185

-- Define the lines and point A
def l₁ (t : ℝ) : ℝ × ℝ := (1 + 3*t, 2 - 4*t)
def l₂ (x y : ℝ) : Prop := 2*x - 4*y = 5
def A : ℝ × ℝ := (1, 2)

-- State the theorem
theorem intersection_distance :
  ∃ (t : ℝ), 
    let B := l₁ t
    l₂ B.1 B.2 ∧ 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l2061_206185


namespace NUMINAMATH_CALUDE_focus_coordinates_l2061_206161

-- Define the ellipse type
structure Ellipse where
  major_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)
  minor_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)

-- Define the function to find the focus with lesser y-coordinate
def focus_with_lesser_y (e : Ellipse) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem focus_coordinates (e : Ellipse) :
  e.major_axis_endpoints = ((1, -2), (7, -2)) →
  e.minor_axis_endpoints = ((4, 1), (4, -5)) →
  focus_with_lesser_y e = (4, -2) :=
sorry

end NUMINAMATH_CALUDE_focus_coordinates_l2061_206161


namespace NUMINAMATH_CALUDE_volume_maximized_at_ten_l2061_206121

/-- The volume of the container as a function of the side length of the cut squares -/
def volume (x : ℝ) : ℝ := (90 - 2*x) * (48 - 2*x) * x

/-- The derivative of the volume function -/
def volume_derivative (x : ℝ) : ℝ := 12*x^2 - 552*x + 4320

theorem volume_maximized_at_ten :
  ∃ (x : ℝ), x > 0 ∧ x < 24 ∧
  (∀ (y : ℝ), y > 0 → y < 24 → volume y ≤ volume x) ∧
  x = 10 := by sorry

end NUMINAMATH_CALUDE_volume_maximized_at_ten_l2061_206121


namespace NUMINAMATH_CALUDE_expression_value_l2061_206111

theorem expression_value : 
  (2015^3 - 3 * 2015^2 * 2016 + 3 * 2015 * 2016^2 - 2016^3 + 1) / (2015 * 2016) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2061_206111


namespace NUMINAMATH_CALUDE_product_of_three_integers_l2061_206147

theorem product_of_three_integers (A B C : Int) : 
  A < B → B < C → A + B + C = 33 → C = 3 * B → A = C - 23 → A * B * C = 192 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_integers_l2061_206147


namespace NUMINAMATH_CALUDE_solution1_composition_l2061_206176

/-- Represents a solution with lemonade and carbonated water -/
structure Solution :=
  (lemonade : ℝ)
  (carbonated_water : ℝ)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution1 : Solution)
  (solution2 : Solution)
  (proportion1 : ℝ)

/-- Theorem stating the composition of Solution 1 given the mixture properties -/
theorem solution1_composition 
  (s1 : Solution)
  (s2 : Solution)
  (m : Mixture)
  (h1 : s1.lemonade = 20)
  (h2 : s2.lemonade = 45)
  (h3 : s2.carbonated_water = 55)
  (h4 : m.solution1 = s1)
  (h5 : m.solution2 = s2)
  (h6 : m.proportion1 = 20)
  (h7 : m.proportion1 * s1.carbonated_water + (100 - m.proportion1) * s2.carbonated_water = 60 * 100) :
  s1.carbonated_water = 80 := by
  sorry

end NUMINAMATH_CALUDE_solution1_composition_l2061_206176


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l2061_206186

-- Define the circles C₁ and C₂
def C₁ (x y a : ℝ) : Prop := x^2 + y^2 - a = 0
def C₂ (x y a : ℝ) : Prop := x^2 + y^2 + 2*x - 2*a*y + 3 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - 4*y + 5 = 0

-- Define symmetry between circles with respect to a line
def symmetric_circles (C₁ C₂ : (ℝ → ℝ → ℝ → Prop)) (l : (ℝ → ℝ → Prop)) : Prop :=
  ∃ a : ℝ, ∀ x y : ℝ, C₁ x y a ∧ C₂ x y a → l x y

-- Theorem statement
theorem circle_symmetry_line :
  symmetric_circles C₁ C₂ line_l :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l2061_206186


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2061_206168

universe u

def I : Set Char := {'b', 'c', 'd', 'e', 'f'}
def M : Set Char := {'b', 'c', 'f'}
def N : Set Char := {'b', 'd', 'e'}

theorem complement_intersection_theorem :
  (I \ M) ∩ N = {'d', 'e'} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2061_206168


namespace NUMINAMATH_CALUDE_bird_sanctuary_problem_l2061_206133

theorem bird_sanctuary_problem (total : ℝ) (total_positive : total > 0) :
  let geese := 0.2 * total
  let swans := 0.4 * total
  let herons := 0.1 * total
  let ducks := 0.2 * total
  let pigeons := total - (geese + swans + herons + ducks)
  let non_herons := total - herons
  (ducks / non_herons) * 100 = 22.22 := by sorry

end NUMINAMATH_CALUDE_bird_sanctuary_problem_l2061_206133


namespace NUMINAMATH_CALUDE_solution_satisfies_congruences_solution_is_three_digit_solution_is_smallest_l2061_206139

/-- The smallest three-digit number satisfying the given congruences -/
def smallest_solution : ℕ := 230

/-- First congruence condition -/
def cong1 (x : ℕ) : Prop := 5 * x ≡ 15 [ZMOD 10]

/-- Second congruence condition -/
def cong2 (x : ℕ) : Prop := 3 * x + 4 ≡ 7 [ZMOD 8]

/-- Third congruence condition -/
def cong3 (x : ℕ) : Prop := -3 * x + 2 ≡ x [ZMOD 17]

/-- The solution satisfies all congruences -/
theorem solution_satisfies_congruences :
  cong1 smallest_solution ∧ 
  cong2 smallest_solution ∧ 
  cong3 smallest_solution :=
sorry

/-- The solution is a three-digit number -/
theorem solution_is_three_digit :
  100 ≤ smallest_solution ∧ smallest_solution < 1000 :=
sorry

/-- The solution is the smallest such number -/
theorem solution_is_smallest (n : ℕ) :
  (100 ≤ n ∧ n < smallest_solution) →
  ¬(cong1 n ∧ cong2 n ∧ cong3 n) :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_congruences_solution_is_three_digit_solution_is_smallest_l2061_206139


namespace NUMINAMATH_CALUDE_five_dice_not_same_probability_l2061_206108

theorem five_dice_not_same_probability :
  let n := 8  -- number of sides on each die
  let k := 5  -- number of dice rolled
  (1 - (n : ℚ) / n^k) = 4095 / 4096 :=
by sorry

end NUMINAMATH_CALUDE_five_dice_not_same_probability_l2061_206108


namespace NUMINAMATH_CALUDE_parallelogram_height_l2061_206166

theorem parallelogram_height (b h_t : ℝ) (h_t_pos : h_t > 0) :
  let a_t := b * h_t / 2
  let h_p := h_t / 2
  let a_p := b * h_p
  h_t = 10 → a_t = a_p → h_p = 5 := by sorry

end NUMINAMATH_CALUDE_parallelogram_height_l2061_206166


namespace NUMINAMATH_CALUDE_sum_of_odd_divisors_180_l2061_206175

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry

theorem sum_of_odd_divisors_180 : sum_of_odd_divisors 180 = 78 := by sorry

end NUMINAMATH_CALUDE_sum_of_odd_divisors_180_l2061_206175


namespace NUMINAMATH_CALUDE_heine_valentine_treats_l2061_206170

/-- Given a total number of biscuits and dogs, calculate the number of biscuits per dog -/
def biscuits_per_dog (total_biscuits : ℕ) (num_dogs : ℕ) : ℕ :=
  total_biscuits / num_dogs

/-- Theorem: Mrs. Heine's Valentine's Day treats distribution -/
theorem heine_valentine_treats :
  let total_biscuits := 6
  let num_dogs := 2
  biscuits_per_dog total_biscuits num_dogs = 3 := by
  sorry

end NUMINAMATH_CALUDE_heine_valentine_treats_l2061_206170


namespace NUMINAMATH_CALUDE_equation_solution_l2061_206140

theorem equation_solution : 
  ∃! x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ∧ x = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2061_206140


namespace NUMINAMATH_CALUDE_circular_table_theorem_l2061_206150

/-- Represents the setup of people sitting around a circular table -/
structure CircularTable where
  num_men : ℕ
  num_women : ℕ

/-- Defines when a man is considered satisfied -/
def is_satisfied (t : CircularTable) : Prop :=
  ∃ (i j : ℕ), i ≠ j ∧ i < t.num_men + t.num_women ∧ j < t.num_men + t.num_women

/-- The probability of a specific man being satisfied -/
def prob_man_satisfied (t : CircularTable) : ℚ :=
  25 / 33

/-- The expected number of satisfied men -/
def expected_satisfied_men (t : CircularTable) : ℚ :=
  1250 / 33

/-- Main theorem statement -/
theorem circular_table_theorem (t : CircularTable) 
  (h1 : t.num_men = 50) (h2 : t.num_women = 50) : 
  prob_man_satisfied t = 25 / 33 ∧ 
  expected_satisfied_men t = 1250 / 33 := by
  sorry

#check circular_table_theorem

end NUMINAMATH_CALUDE_circular_table_theorem_l2061_206150


namespace NUMINAMATH_CALUDE_three_zeros_condition_l2061_206127

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- Theorem stating the condition for f to have exactly 3 zeros -/
theorem three_zeros_condition (a : ℝ) : 
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔ 
  a < -3 ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_condition_l2061_206127


namespace NUMINAMATH_CALUDE_maya_total_pages_l2061_206114

def maya_reading_problem (books_last_week : ℕ) (pages_per_book : ℕ) (reading_increase : ℕ) : ℕ :=
  let pages_last_week := books_last_week * pages_per_book
  let pages_this_week := reading_increase * pages_last_week
  pages_last_week + pages_this_week

theorem maya_total_pages :
  maya_reading_problem 5 300 2 = 4500 :=
by
  sorry

end NUMINAMATH_CALUDE_maya_total_pages_l2061_206114


namespace NUMINAMATH_CALUDE_greyson_payment_l2061_206152

/-- The number of dimes in a dollar -/
def dimes_per_dollar : ℕ := 10

/-- The cost of the watch in dollars -/
def watch_cost : ℕ := 5

/-- The number of dimes Greyson paid for the watch -/
def dimes_paid : ℕ := watch_cost * dimes_per_dollar

theorem greyson_payment : dimes_paid = 50 := by
  sorry

end NUMINAMATH_CALUDE_greyson_payment_l2061_206152


namespace NUMINAMATH_CALUDE_sum_of_abc_l2061_206125

theorem sum_of_abc (a b c : ℕ+) 
  (h1 : (a + b + c)^3 - a^3 - b^3 - c^3 = 150)
  (h2 : a < b) (h3 : b < c) : 
  a + b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l2061_206125


namespace NUMINAMATH_CALUDE_fraction_above_line_is_seven_tenths_l2061_206163

/-- A square in the 2D plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A line in the 2D plane defined by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculate the fraction of a square's area above a given line -/
def fractionAboveLine (s : Square) (l : Line) : ℝ :=
  sorry

/-- The main theorem stating that the fraction of the square's area above the specified line is 7/10 -/
theorem fraction_above_line_is_seven_tenths :
  let s : Square := { bottomLeft := (2, 0), topRight := (7, 5) }
  let l : Line := { point1 := (2, 1), point2 := (7, 3) }
  fractionAboveLine s l = 7/10 := by sorry

end NUMINAMATH_CALUDE_fraction_above_line_is_seven_tenths_l2061_206163


namespace NUMINAMATH_CALUDE_circle_radius_zero_l2061_206192

/-- The radius of the circle described by the equation 4x^2 - 8x + 4y^2 - 16y + 20 = 0 is 0 -/
theorem circle_radius_zero (x y : ℝ) : 
  (4*x^2 - 8*x + 4*y^2 - 16*y + 20 = 0) → 
  ∃ (h k : ℝ), ∀ (x y : ℝ), 4*x^2 - 8*x + 4*y^2 - 16*y + 20 = 0 ↔ (x - h)^2 + (y - k)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l2061_206192


namespace NUMINAMATH_CALUDE_wendy_photo_albums_l2061_206153

theorem wendy_photo_albums 
  (phone_pics camera_pics pics_per_album : ℕ) 
  (h1 : phone_pics = 22)
  (h2 : camera_pics = 2)
  (h3 : pics_per_album = 6)
  : (phone_pics + camera_pics) / pics_per_album = 4 := by
  sorry

end NUMINAMATH_CALUDE_wendy_photo_albums_l2061_206153


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_theorem_l2061_206182

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (coplanar : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_theorem 
  (m n : Line) (α : Plane) 
  (h1 : subset m α) 
  (h2 : parallel n α) 
  (h3 : coplanar m n) : 
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_theorem_l2061_206182


namespace NUMINAMATH_CALUDE_negative_division_equals_positive_division_of_negative_integers_l2061_206184

theorem negative_division_equals_positive (a b : Int) (h : b ≠ 0) :
  (-a) / (-b) = a / b :=
sorry

theorem division_of_negative_integers :
  (-81) / (-9) = 9 :=
sorry

end NUMINAMATH_CALUDE_negative_division_equals_positive_division_of_negative_integers_l2061_206184


namespace NUMINAMATH_CALUDE_triangle_problem_l2061_206171

open Real

theorem triangle_problem (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  sin C * sin (A - B) = sin B * sin (C - A) ∧
  A = 2 * B →
  C = 5 * π / 8 ∧ 2 * a^2 = b^2 + c^2 :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2061_206171


namespace NUMINAMATH_CALUDE_function_properties_l2061_206158

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period_one_negation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f x

theorem function_properties (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_period_neg : has_period_one_negation f) : 
  (∀ x, f (x + 2) = f x) ∧ 
  (∀ (x : ℝ) (k : ℤ), f (2 * ↑k - x) = -f x) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2061_206158


namespace NUMINAMATH_CALUDE_eighteenth_term_of_equal_sum_sequence_l2061_206110

/-- An equal sum sequence is a sequence where the sum of each term and its next term is constant. -/
def EqualSumSequence (a : ℕ → ℝ) (sum : ℝ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = sum

theorem eighteenth_term_of_equal_sum_sequence
  (a : ℕ → ℝ)
  (sum : ℝ)
  (h_equal_sum : EqualSumSequence a sum)
  (h_first_term : a 1 = 2)
  (h_sum : sum = 5) :
  a 18 = 3 := by
sorry

end NUMINAMATH_CALUDE_eighteenth_term_of_equal_sum_sequence_l2061_206110


namespace NUMINAMATH_CALUDE_x_power_six_plus_reciprocal_l2061_206131

theorem x_power_six_plus_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^6 + 1/x^6 = 322 := by
  sorry

end NUMINAMATH_CALUDE_x_power_six_plus_reciprocal_l2061_206131


namespace NUMINAMATH_CALUDE_smallest_angle_greater_than_36_degrees_l2061_206181

/-- Represents the angles of a convex pentagon in arithmetic progression -/
structure ConvexPentagonAngles where
  α : ℝ  -- smallest angle
  γ : ℝ  -- common difference
  convex : α > 0 ∧ γ ≥ 0 ∧ α + 4*γ < π  -- convexity condition
  sum : α + (α + γ) + (α + 2*γ) + (α + 3*γ) + (α + 4*γ) = 3*π  -- sum of angles

/-- 
Theorem: In a convex pentagon with angles in arithmetic progression, 
the smallest angle is greater than π/5 radians (36 degrees).
-/
theorem smallest_angle_greater_than_36_degrees (p : ConvexPentagonAngles) : 
  p.α > π/5 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_greater_than_36_degrees_l2061_206181


namespace NUMINAMATH_CALUDE_line_l_equation_line_l₃_equation_l2061_206128

-- Define the point M
def M : ℝ × ℝ := (3, 0)

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y - 2 = 0
def l₂ (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the line l
def l (x y : ℝ) : Prop := 8 * x - y - 24 = 0

-- Define the line l₃
def l₃ (x y : ℝ) : Prop := x - 2 * y - 5 = 0

-- Theorem for the equation of line l
theorem line_l_equation : 
  ∃ (P Q : ℝ × ℝ), 
    l₁ P.1 P.2 ∧ 
    l₂ Q.1 Q.2 ∧ 
    l M.1 M.2 ∧ 
    l P.1 P.2 ∧ 
    l Q.1 Q.2 ∧ 
    M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) :=
sorry

-- Theorem for the equation of line l₃
theorem line_l₃_equation :
  ∃ (I : ℝ × ℝ), 
    l₁ I.1 I.2 ∧ 
    l₂ I.1 I.2 ∧ 
    l₃ I.1 I.2 ∧
    ∀ (x y : ℝ), l₃ x y ↔ x - 2 * y - 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_line_l₃_equation_l2061_206128


namespace NUMINAMATH_CALUDE_det_2x2_matrix_l2061_206173

/-- The determinant of a 2x2 matrix [[x, 4], [-3, y]] is xy + 12 -/
theorem det_2x2_matrix (x y : ℝ) : 
  Matrix.det !![x, 4; -3, y] = x * y + 12 := by
  sorry

end NUMINAMATH_CALUDE_det_2x2_matrix_l2061_206173


namespace NUMINAMATH_CALUDE_optimal_parking_allocation_l2061_206146

/-- Represents the parking space allocation problem -/
structure ParkingAllocation where
  total_spaces : ℕ
  above_ground_cost : ℚ
  underground_cost : ℚ
  total_budget : ℚ
  above_ground_spaces : ℕ
  underground_spaces : ℕ

/-- The optimal allocation satisfies the given conditions -/
def is_optimal_allocation (p : ParkingAllocation) : Prop :=
  p.total_spaces = 5000 ∧
  3 * p.above_ground_cost + 2 * p.underground_cost = 0.8 ∧
  2 * p.above_ground_cost + 4 * p.underground_cost = 1.2 ∧
  p.above_ground_cost = 0.1 ∧
  p.underground_cost = 0.25 ∧
  p.total_budget = 950 ∧
  p.above_ground_spaces + p.underground_spaces = p.total_spaces ∧
  p.above_ground_spaces * p.above_ground_cost + p.underground_spaces * p.underground_cost ≤ p.total_budget

/-- The theorem stating the optimal allocation -/
theorem optimal_parking_allocation :
  ∃ (p : ParkingAllocation), is_optimal_allocation p ∧ 
    p.above_ground_spaces = 2000 ∧ p.underground_spaces = 3000 := by
  sorry


end NUMINAMATH_CALUDE_optimal_parking_allocation_l2061_206146


namespace NUMINAMATH_CALUDE_coat_cost_proof_l2061_206145

/-- The cost of the more expensive coat -/
def expensive_coat_cost : ℝ := 300

/-- The lifespan of the more expensive coat in years -/
def expensive_coat_lifespan : ℕ := 15

/-- The cost of the cheaper coat -/
def cheaper_coat_cost : ℝ := 120

/-- The lifespan of the cheaper coat in years -/
def cheaper_coat_lifespan : ℕ := 5

/-- The number of years over which we compare the costs -/
def comparison_period : ℕ := 30

/-- The amount saved by buying the more expensive coat over the comparison period -/
def savings : ℝ := 120

theorem coat_cost_proof :
  expensive_coat_cost * (comparison_period / expensive_coat_lifespan) =
  cheaper_coat_cost * (comparison_period / cheaper_coat_lifespan) - savings :=
by sorry

end NUMINAMATH_CALUDE_coat_cost_proof_l2061_206145


namespace NUMINAMATH_CALUDE_smallest_special_number_l2061_206148

theorem smallest_special_number : ∃ (n : ℕ), 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (∃ (k : ℕ), n = 2 * k) ∧
  (∃ (k : ℕ), n + 1 = 3 * k) ∧
  (∃ (k : ℕ), n + 2 = 4 * k) ∧
  (∃ (k : ℕ), n + 3 = 5 * k) ∧
  (∃ (k : ℕ), n + 4 = 6 * k) ∧
  (∀ (m : ℕ), 
    (100 ≤ m ∧ m < n) →
    (¬(∃ (k : ℕ), m = 2 * k) ∨
     ¬(∃ (k : ℕ), m + 1 = 3 * k) ∨
     ¬(∃ (k : ℕ), m + 2 = 4 * k) ∨
     ¬(∃ (k : ℕ), m + 3 = 5 * k) ∨
     ¬(∃ (k : ℕ), m + 4 = 6 * k))) ∧
  n = 122 :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_number_l2061_206148


namespace NUMINAMATH_CALUDE_kirill_height_l2061_206156

/-- Represents the heights of Kirill, his brother, sister, and cousin --/
structure FamilyHeights where
  kirill : ℕ
  brother : ℕ
  sister : ℕ
  cousin : ℕ

/-- The conditions of the problem --/
def HeightConditions (h : FamilyHeights) : Prop :=
  h.brother = h.kirill + 14 ∧
  h.sister = 2 * h.kirill ∧
  h.cousin = h.sister + 3 ∧
  h.kirill + h.brother + h.sister + h.cousin = 432

/-- The theorem stating Kirill's height --/
theorem kirill_height :
  ∀ h : FamilyHeights, HeightConditions h → h.kirill = 69 :=
by sorry

end NUMINAMATH_CALUDE_kirill_height_l2061_206156


namespace NUMINAMATH_CALUDE_farm_heads_count_l2061_206129

theorem farm_heads_count (num_hens : ℕ) (total_feet : ℕ) : 
  num_hens = 30 → total_feet = 140 → 
  ∃ (num_cows : ℕ), 
    num_hens + num_cows = 50 ∧
    num_hens * 2 + num_cows * 4 = total_feet :=
by sorry

end NUMINAMATH_CALUDE_farm_heads_count_l2061_206129


namespace NUMINAMATH_CALUDE_square_value_l2061_206130

theorem square_value (square x : ℤ) 
  (h1 : square + x = 80)
  (h2 : 3 * (square + x) - 2 * x = 164) : 
  square = 42 := by
sorry

end NUMINAMATH_CALUDE_square_value_l2061_206130


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l2061_206191

/-- Given two hyperbolas with equations x²/9 - y²/16 = 1 and y²/25 - x²/N = 1,
    if they have the same asymptotes, then N = 225/16 -/
theorem hyperbolas_same_asymptotes (N : ℝ) :
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / N = 1) →
  N = 225 / 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l2061_206191


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l2061_206155

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (y : ℕ), y = Nat.lcm x (Nat.lcm 15 21) ∧ y = 105) →
  x ≤ 105 ∧ ∃ (z : ℕ), z = 105 ∧ (∃ (w : ℕ), w = Nat.lcm z (Nat.lcm 15 21) ∧ w = 105) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l2061_206155


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2061_206159

-- Define the fraction 7/12
def fraction : ℚ := 7 / 12

-- Define the decimal approximation
def decimal_approx : ℝ := 0.5833

-- Define the maximum allowed error due to rounding
def max_error : ℝ := 0.00005

-- Theorem statement
theorem fraction_to_decimal :
  |((fraction : ℝ) - decimal_approx)| < max_error := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2061_206159


namespace NUMINAMATH_CALUDE_quadratic_root_property_l2061_206174

theorem quadratic_root_property (a : ℝ) : 
  2 * a^2 + 3 * a - 2022 = 0 → 2 - 6 * a - 4 * a^2 = -4042 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l2061_206174


namespace NUMINAMATH_CALUDE_circle_equation_l2061_206172

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line L: x - y - 1 = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 - 1 = 0}

-- Define points A and B
def A : ℝ × ℝ := (4, 1)
def B : ℝ × ℝ := (2, 1)

-- State the theorem
theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    -- The circle passes through point A
    A ∈ Circle center radius ∧
    -- The circle is tangent to the line at point B
    B ∈ Circle center radius ∧
    B ∈ Line ∧
    -- The equation of the circle is (x-3)²+y²=2
    center = (3, 0) ∧ radius^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2061_206172


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_l2061_206199

def product : ℕ := 11 * 101 * 111 * 110011

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_product :
  sum_of_digits product = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_product_l2061_206199


namespace NUMINAMATH_CALUDE_base_7_326_equals_base_4_2213_l2061_206115

def base_7_to_decimal (x y z : ℕ) : ℕ := x * 7^2 + y * 7 + z

def decimal_to_base_4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base_7_326_equals_base_4_2213 :
  decimal_to_base_4 (base_7_to_decimal 3 2 6) = [2, 2, 1, 3] := by
  sorry

end NUMINAMATH_CALUDE_base_7_326_equals_base_4_2213_l2061_206115


namespace NUMINAMATH_CALUDE_area_of_r3_l2061_206102

/-- Given a square R1 with area 36, R2 formed by connecting midpoints of R1's sides,
    and R3 formed by moving R2's corners halfway to its center, prove R3's area is 4.5 -/
theorem area_of_r3 (r1 r2 r3 : Real) : 
  r1^2 = 36 → 
  r2 = r1 * Real.sqrt 2 / 2 → 
  r3 = r2 / 2 → 
  r3^2 = 4.5 := by sorry

end NUMINAMATH_CALUDE_area_of_r3_l2061_206102


namespace NUMINAMATH_CALUDE_winning_strategy_correct_l2061_206157

/-- A stone-picking game with two players. -/
structure StoneGame where
  total_stones : ℕ
  min_take : ℕ
  max_take : ℕ

/-- A winning strategy for the first player in the stone game. -/
def winning_first_move (game : StoneGame) : ℕ := 3

/-- Theorem stating that the winning strategy for the first player is correct. -/
theorem winning_strategy_correct (game : StoneGame) 
  (h1 : game.total_stones = 18)
  (h2 : game.min_take = 1)
  (h3 : game.max_take = 4) :
  ∃ (n : ℕ), n ≥ game.min_take ∧ n ≤ game.max_take ∧
  (winning_first_move game = n → 
   ∀ (m : ℕ), m ≥ game.min_take → m ≤ game.max_take → 
   ∃ (k : ℕ), k ≥ game.min_take ∧ k ≤ game.max_take ∧
   (game.total_stones - n - m - k) % 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_winning_strategy_correct_l2061_206157


namespace NUMINAMATH_CALUDE_college_choice_probability_l2061_206100

theorem college_choice_probability : 
  let num_examinees : ℕ := 2
  let num_colleges : ℕ := 3
  let prob_choose_college : ℚ := 1 / num_colleges
  
  -- Probability that both examinees choose the third college
  let prob_both_choose_third : ℚ := prob_choose_college ^ num_examinees
  
  -- Probability that at least one of the first two colleges is chosen
  let prob_at_least_one_first_two : ℚ := 1 - prob_both_choose_third
  
  prob_at_least_one_first_two = 8 / 9 :=
by sorry

end NUMINAMATH_CALUDE_college_choice_probability_l2061_206100


namespace NUMINAMATH_CALUDE_x_is_28_percent_greater_than_150_l2061_206103

theorem x_is_28_percent_greater_than_150 :
  ∀ x : ℝ, x = 150 * (1 + 28/100) → x = 192 := by
  sorry

end NUMINAMATH_CALUDE_x_is_28_percent_greater_than_150_l2061_206103


namespace NUMINAMATH_CALUDE_werewolf_identity_l2061_206144

structure Person where
  name : String
  is_knight : Bool
  is_werewolf : Bool
  is_liar : Bool

def A : Person := { name := "A", is_knight := true, is_werewolf := true, is_liar := false }
def B : Person := { name := "B", is_knight := false, is_werewolf := false, is_liar := true }
def C : Person := { name := "C", is_knight := false, is_werewolf := false, is_liar := true }

theorem werewolf_identity (A B C : Person) :
  (A.is_knight ↔ (A.is_liar ∨ B.is_liar ∨ C.is_liar)) →
  (B.is_knight ↔ C.is_knight) →
  ((A.is_werewolf ∧ A.is_knight) ∨ (B.is_werewolf ∧ B.is_knight)) →
  (A.is_werewolf ∨ B.is_werewolf) →
  ¬(A.is_werewolf ∧ B.is_werewolf) →
  A.is_werewolf := by
  sorry

#check werewolf_identity A B C

end NUMINAMATH_CALUDE_werewolf_identity_l2061_206144


namespace NUMINAMATH_CALUDE_first_day_pen_sales_l2061_206126

/-- Proves that given the conditions of pen sales over 13 days, 
    the number of pens sold on the first day is 96. -/
theorem first_day_pen_sales : ∀ (first_day_sales : ℕ),
  (first_day_sales + 12 * 44 = 13 * 48) →
  first_day_sales = 96 := by
  sorry

#check first_day_pen_sales

end NUMINAMATH_CALUDE_first_day_pen_sales_l2061_206126


namespace NUMINAMATH_CALUDE_largest_common_divisor_420_385_l2061_206113

def largest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem largest_common_divisor_420_385 :
  largest_common_divisor 420 385 = 35 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_420_385_l2061_206113


namespace NUMINAMATH_CALUDE_son_age_l2061_206149

theorem son_age (son_age man_age : ℕ) : 
  man_age = son_age + 25 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 23 := by
sorry

end NUMINAMATH_CALUDE_son_age_l2061_206149


namespace NUMINAMATH_CALUDE_sixPeopleRoundTable_l2061_206177

/-- Number of distinct seating arrangements for n people around a round table,
    considering rotational symmetry -/
def roundTableSeating (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of distinct seating arrangements for 6 people around a round table,
    considering rotational symmetry, is 120 -/
theorem sixPeopleRoundTable : roundTableSeating 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_sixPeopleRoundTable_l2061_206177


namespace NUMINAMATH_CALUDE_speech_contest_probabilities_l2061_206193

def num_boys : ℕ := 4
def num_girls : ℕ := 2
def num_selected : ℕ := 2

def total_combinations : ℕ := (num_boys + num_girls).choose num_selected

theorem speech_contest_probabilities :
  let p_two_boys := (num_boys.choose num_selected : ℚ) / total_combinations
  let p_one_girl := (num_boys.choose 1 * num_girls.choose 1 : ℚ) / total_combinations
  let p_at_least_one_girl := 1 - p_two_boys
  (p_two_boys = 2/5) ∧
  (p_one_girl = 8/15) ∧
  (p_at_least_one_girl = 3/5) := by sorry

end NUMINAMATH_CALUDE_speech_contest_probabilities_l2061_206193


namespace NUMINAMATH_CALUDE_root_implies_m_minus_n_l2061_206190

theorem root_implies_m_minus_n (m n : ℝ) : 
  ((-3)^2 + m*(-3) + 3*n = 0) → (m - n = 3) := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_minus_n_l2061_206190


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2061_206183

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x > 3 ∨ x < -1}

-- Define set B
def B : Set ℝ := {x | x > 0}

-- State the theorem
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {x : ℝ | 0 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2061_206183


namespace NUMINAMATH_CALUDE_sum_x_y_equals_one_third_l2061_206143

theorem sum_x_y_equals_one_third 
  (x y a : ℚ) 
  (eq1 : 17 * x + 19 * y = 6 - a) 
  (eq2 : 13 * x - 7 * y = 10 * a + 1) : 
  x + y = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_one_third_l2061_206143


namespace NUMINAMATH_CALUDE_pizza_calculation_l2061_206116

/-- The total number of pizzas made by Heather and Craig in two days -/
def total_pizzas (craig_day1 : ℕ) (heather_multiplier : ℕ) (craig_increase : ℕ) (heather_decrease : ℕ) : ℕ :=
  let craig_day2 := craig_day1 + craig_increase
  let heather_day1 := craig_day1 * heather_multiplier
  let heather_day2 := craig_day2 - heather_decrease
  craig_day1 + heather_day1 + craig_day2 + heather_day2

/-- Theorem stating the total number of pizzas made by Heather and Craig in two days -/
theorem pizza_calculation : total_pizzas 40 4 60 20 = 380 := by
  sorry

end NUMINAMATH_CALUDE_pizza_calculation_l2061_206116


namespace NUMINAMATH_CALUDE_fruit_arrangement_l2061_206162

-- Define the fruits and boxes
inductive Fruit
| Apple
| Pear
| Orange
| Banana

inductive Box
| One
| Two
| Three
| Four

-- Define the arrangement of fruits in boxes
def Arrangement := Box → Fruit

-- Define the labels on the boxes
def Label (b : Box) : Prop :=
  match b with
  | Box.One => ∃ a : Arrangement, a Box.One = Fruit.Orange
  | Box.Two => ∃ a : Arrangement, a Box.Two = Fruit.Pear
  | Box.Three => ∃ a : Arrangement, (a Box.One = Fruit.Banana) → (a Box.Three = Fruit.Apple ∨ a Box.Three = Fruit.Pear)
  | Box.Four => ∃ a : Arrangement, a Box.Four = Fruit.Apple

-- Define the correct arrangement
def CorrectArrangement : Arrangement :=
  fun b => match b with
  | Box.One => Fruit.Banana
  | Box.Two => Fruit.Apple
  | Box.Three => Fruit.Orange
  | Box.Four => Fruit.Pear

-- State the theorem
theorem fruit_arrangement :
  ∀ a : Arrangement,
  (∀ b : Box, ∀ f : Fruit, (a b = f) → (∃! b' : Box, a b' = f)) →
  (∀ b : Box, ¬Label b) →
  a = CorrectArrangement :=
sorry

end NUMINAMATH_CALUDE_fruit_arrangement_l2061_206162
