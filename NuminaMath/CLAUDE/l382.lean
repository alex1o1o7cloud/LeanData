import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_factorization_l382_38283

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 2*x) * (x^2 + 2*x + 2) + 1 = (x + 1)^4 ∧
  (x^2 - 4*x) * (x^2 - 4*x + 8) + 16 = (x - 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l382_38283


namespace NUMINAMATH_CALUDE_expression_evaluation_l382_38284

theorem expression_evaluation (b : ℚ) (h : b = 4/3) :
  (3 * b^2 - 14 * b + 5) * (3 * b - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l382_38284


namespace NUMINAMATH_CALUDE_like_term_proof_l382_38269

def is_like_term (t₁ t₂ : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ), t₁ x y = a * x^5 * y^3 ∧ t₂ x y = b * x^5 * y^3

theorem like_term_proof (a : ℝ) :
  is_like_term (λ x y => -5 * x^5 * y^3) (λ x y => a * x^5 * y^3) := by
  sorry

end NUMINAMATH_CALUDE_like_term_proof_l382_38269


namespace NUMINAMATH_CALUDE_percentage_reading_both_books_l382_38281

theorem percentage_reading_both_books (total_students : ℕ) 
  (read_A : ℕ) (read_B : ℕ) (read_both : ℕ) :
  total_students = 600 →
  read_both = (20 * read_A) / 100 →
  read_A + read_B - read_both = total_students →
  read_A - read_both - (read_B - read_both) = 75 →
  (read_both * 100) / read_B = 25 :=
by sorry

end NUMINAMATH_CALUDE_percentage_reading_both_books_l382_38281


namespace NUMINAMATH_CALUDE_mean_height_is_correct_l382_38203

/-- Represents the heights of players on a basketball team -/
def heights : List Nat := [57, 62, 64, 64, 65, 67, 68, 70, 71, 72, 72, 73, 74, 75, 75]

/-- The number of players on the team -/
def num_players : Nat := heights.length

/-- The sum of all player heights -/
def total_height : Nat := heights.sum

/-- Calculates the mean height of the players -/
def mean_height : Rat := total_height / num_players

theorem mean_height_is_correct : mean_height = 1029 / 15 := by sorry

end NUMINAMATH_CALUDE_mean_height_is_correct_l382_38203


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l382_38227

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- State the theorem
theorem set_intersection_theorem :
  M ∩ (Set.univ \ N) = {x : ℝ | -2 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l382_38227


namespace NUMINAMATH_CALUDE_intersection_equality_l382_38202

theorem intersection_equality (m : ℝ) : 
  ({3, 4, m^2 - 3*m - 1} ∩ {2*m, -3} : Set ℝ) = {-3} → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l382_38202


namespace NUMINAMATH_CALUDE_three_numbers_sum_l382_38200

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ordering of numbers
  b = 10 →  -- Median is 10
  (a + b + c) / 3 = a + 8 →  -- Mean is 8 more than least
  (a + b + c) / 3 = c - 20 →  -- Mean is 20 less than greatest
  a + b + c = 66 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l382_38200


namespace NUMINAMATH_CALUDE_final_distance_calculation_l382_38266

/-- Calculates the final distance between two cars given their initial conditions and speed changes --/
theorem final_distance_calculation (initial_speed initial_distance city_speed good_road_speed dirt_road_speed : ℝ) :
  initial_speed > 0 ∧ 
  initial_distance > 0 ∧ 
  city_speed > 0 ∧ 
  good_road_speed > 0 ∧ 
  dirt_road_speed > 0 →
  initial_speed = 60 ∧ 
  initial_distance = 2 ∧ 
  city_speed = 40 ∧ 
  good_road_speed = 70 ∧ 
  dirt_road_speed = 30 →
  initial_distance * (city_speed / initial_speed) * (good_road_speed / city_speed) * (dirt_road_speed / good_road_speed) = 1 := by
  sorry

#check final_distance_calculation

end NUMINAMATH_CALUDE_final_distance_calculation_l382_38266


namespace NUMINAMATH_CALUDE_new_average_age_l382_38233

theorem new_average_age (initial_people : ℕ) (initial_avg : ℚ) (leaving_age : ℕ) (entering_age : ℕ) :
  initial_people = 7 →
  initial_avg = 28 →
  leaving_age = 22 →
  entering_age = 30 →
  round ((initial_people * initial_avg - leaving_age + entering_age) / initial_people) = 29 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l382_38233


namespace NUMINAMATH_CALUDE_sqrt_neg_nine_squared_l382_38216

theorem sqrt_neg_nine_squared : Real.sqrt ((-9)^2) = 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_neg_nine_squared_l382_38216


namespace NUMINAMATH_CALUDE_cubic_function_property_l382_38247

/-- Given a cubic function f(x) = mx³ + nx + 1 where mn ≠ 0 and f(-1) = 5, prove that f(1) = 7 -/
theorem cubic_function_property (m n : ℝ) (h1 : m * n ≠ 0) :
  let f := fun x : ℝ => m * x^3 + n * x + 1
  f (-1) = 5 → f 1 = 7 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l382_38247


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_y_axis_l382_38232

/-- Given a point A with coordinates (-2, 4), this theorem proves that the point
    symmetric to A with respect to the y-axis has coordinates (2, 4). -/
theorem symmetric_point_wrt_y_axis :
  let A : ℝ × ℝ := (-2, 4)
  let symmetric_point := (-(A.1), A.2)
  symmetric_point = (2, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_y_axis_l382_38232


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l382_38223

/-- Given a line ax + by + c = 0 where ac < 0 and bc < 0, prove that the line does not pass through the third quadrant. -/
theorem line_not_in_third_quadrant (a b c : ℝ) (h1 : a * c < 0) (h2 : b * c < 0) :
  ¬∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l382_38223


namespace NUMINAMATH_CALUDE_subtract_negative_l382_38290

theorem subtract_negative : 2 - (-3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l382_38290


namespace NUMINAMATH_CALUDE_scooter_repair_percentage_l382_38296

theorem scooter_repair_percentage (profit_percentage : ℝ) (profit_amount : ℝ) (repair_cost : ℝ) :
  profit_percentage = 0.2 →
  profit_amount = 1100 →
  repair_cost = 500 →
  (repair_cost / (profit_amount / profit_percentage)) * 100 = 500 / 5500 * 100 := by
sorry

end NUMINAMATH_CALUDE_scooter_repair_percentage_l382_38296


namespace NUMINAMATH_CALUDE_max_shapes_in_grid_l382_38221

/-- The number of rows in the grid -/
def rows : Nat := 8

/-- The number of columns in the grid -/
def columns : Nat := 14

/-- The number of grid points occupied by each shape -/
def points_per_shape : Nat := 8

/-- The total number of grid points in the grid -/
def total_grid_points : Nat := (rows + 1) * (columns + 1)

/-- The maximum number of shapes that can be placed in the grid -/
def max_shapes : Nat := total_grid_points / points_per_shape

theorem max_shapes_in_grid :
  max_shapes = 16 := by sorry

end NUMINAMATH_CALUDE_max_shapes_in_grid_l382_38221


namespace NUMINAMATH_CALUDE_parabola_transformation_l382_38201

/-- Represents a parabola of the form y = (x + a)^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Applies a horizontal shift to a parabola -/
def horizontal_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a - shift, b := p.b }

/-- Applies a vertical shift to a parabola -/
def vertical_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, b := p.b + shift }

/-- The initial parabola y = (x + 2)^2 + 3 -/
def initial_parabola : Parabola := { a := 2, b := 3 }

/-- The final parabola after transformations -/
def final_parabola : Parabola := { a := -1, b := 1 }

theorem parabola_transformation :
  (vertical_shift (horizontal_shift initial_parabola 3) (-2)) = final_parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l382_38201


namespace NUMINAMATH_CALUDE_equation_is_parabola_l382_38293

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents the equation |y - 3| = √((x+4)² + (y-1)²) -/
def equation (p : Point2D) : Prop :=
  |p.y - 3| = Real.sqrt ((p.x + 4)^2 + (p.y - 1)^2)

/-- Represents a parabola in general form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point satisfies the parabola equation -/
def satisfies_parabola (p : Point2D) (para : Parabola) : Prop :=
  p.y = para.a * p.x^2 + para.b * p.x + para.c

/-- Theorem stating that the given equation represents a parabola -/
theorem equation_is_parabola :
  ∃ (para : Parabola), ∀ (p : Point2D), equation p → satisfies_parabola p para :=
sorry

end NUMINAMATH_CALUDE_equation_is_parabola_l382_38293


namespace NUMINAMATH_CALUDE_journey_time_theorem_l382_38275

/-- Represents the time and distance relationship for a journey to the supermarket -/
structure JourneyTime where
  bike_speed : ℝ
  walk_speed : ℝ
  total_distance : ℝ

/-- The journey time satisfies the given conditions -/
def satisfies_conditions (j : JourneyTime) : Prop :=
  j.bike_speed * 12 + j.walk_speed * 20 = j.total_distance ∧
  j.bike_speed * 8 + j.walk_speed * 36 = j.total_distance

/-- The theorem to be proved -/
theorem journey_time_theorem (j : JourneyTime) (h : satisfies_conditions j) :
  (j.total_distance - j.bike_speed * 2) / j.walk_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_theorem_l382_38275


namespace NUMINAMATH_CALUDE_rowing_current_velocity_l382_38297

/-- Proves that the velocity of the current is 1 kmph given the conditions of the rowing problem. -/
theorem rowing_current_velocity 
  (still_water_speed : ℝ) 
  (distance : ℝ) 
  (total_time : ℝ) 
  (h1 : still_water_speed = 5)
  (h2 : distance = 2.4)
  (h3 : total_time = 1) :
  ∃ v : ℝ, v = 1 ∧ total_time = distance / (still_water_speed + v) + distance / (still_water_speed - v) :=
by sorry

end NUMINAMATH_CALUDE_rowing_current_velocity_l382_38297


namespace NUMINAMATH_CALUDE_linear_equation_condition_l382_38267

/-- Given that (a-2)x^(|a|-1) + 3y = 1 is a linear equation in x and y, prove that a = -2 -/
theorem linear_equation_condition (a : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, (a - 2) * x^(|a| - 1) + 3 * y = 1 ↔ k * x + 3 * y = 1) ↔ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l382_38267


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l382_38270

/-- Given a quadratic equation x^2 + px + q = 0 whose roots are each three more than
    the roots of 2x^2 - 4x - 5, prove that q = 25/2 -/
theorem quadratic_roots_relation (p q : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ ∃ y, 2*y^2 - 4*y - 5 = 0 ∧ x = y + 3) →
  q = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l382_38270


namespace NUMINAMATH_CALUDE_odd_function_symmetry_l382_38229

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def isDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y
def hasMinimumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop := 
  ∀ x, a ≤ x ∧ x ≤ b → f x ≥ m

-- State the theorem
theorem odd_function_symmetry (hOdd : isOdd f) 
  (hDec : isDecreasingOn f (-2) (-1)) 
  (hMin : hasMinimumOn f (-2) (-1) 3) :
  isDecreasingOn f 1 2 ∧ hasMinimumOn f 1 2 (-3) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_symmetry_l382_38229


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l382_38250

theorem same_remainder_divisor : ∃ (N : ℕ), N > 1 ∧ 
  N = 23 ∧ 
  (1743 % N = 2019 % N) ∧ 
  (2019 % N = 3008 % N) ∧ 
  ∀ (M : ℕ), M > N → (1743 % M ≠ 2019 % M ∨ 2019 % M ≠ 3008 % M) := by
  sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l382_38250


namespace NUMINAMATH_CALUDE_min_value_expression_l382_38288

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 48) :
  x^2 + 4*x*y + 4*y^2 + 3*z^2 ≥ 144 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 48 ∧ x₀^2 + 4*x₀*y₀ + 4*y₀^2 + 3*z₀^2 = 144 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l382_38288


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l382_38276

theorem repeating_decimal_sum (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  (10 * a + b) / 99 + (100 * a + 10 * b + c) / 999 = 12 / 13 →
  a = 4 ∧ b = 6 ∧ c = 3 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l382_38276


namespace NUMINAMATH_CALUDE_real_number_classification_l382_38207

theorem real_number_classification : 
  ∀ x : ℝ, x < 0 ∨ x ≥ 0 := by sorry

end NUMINAMATH_CALUDE_real_number_classification_l382_38207


namespace NUMINAMATH_CALUDE_cayley_hamilton_for_B_l382_38244

def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, 3; 2, 1, 2; 3, 2, 1]

theorem cayley_hamilton_for_B :
  ∃ (s t u : ℝ), 
    B^3 + s • B^2 + t • B + u • (1 : Matrix (Fin 3) (Fin 3) ℝ) = 0 ∧ 
    s = -7 ∧ t = 2 ∧ u = -9 := by
  sorry

end NUMINAMATH_CALUDE_cayley_hamilton_for_B_l382_38244


namespace NUMINAMATH_CALUDE_sequence_solution_l382_38231

def x (n : ℕ+) : ℚ := n / (n + 2016)

theorem sequence_solution :
  ∃ (m n : ℕ+), x 2016 = x m * x n ∧ m = 4032 ∧ n = 6048 :=
by sorry

end NUMINAMATH_CALUDE_sequence_solution_l382_38231


namespace NUMINAMATH_CALUDE_ones_digit_of_6_power_52_l382_38292

theorem ones_digit_of_6_power_52 : ∃ n : ℕ, 6^52 = 10 * n + 6 :=
sorry

end NUMINAMATH_CALUDE_ones_digit_of_6_power_52_l382_38292


namespace NUMINAMATH_CALUDE_subset_sum_exists_l382_38239

theorem subset_sum_exists (nums : List ℕ) : 
  nums.length = 100 → 
  (∀ n ∈ nums, n ≤ 100) → 
  nums.sum = 200 → 
  ∃ subset : List ℕ, subset ⊆ nums ∧ subset.sum = 100 := by
  sorry

end NUMINAMATH_CALUDE_subset_sum_exists_l382_38239


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l382_38215

/-- A line is tangent to a parabola if and only if the resulting quadratic equation has a double root --/
axiom tangent_iff_double_root (a b c : ℝ) : 
  (∃ k, a * k^2 + b * k + c = 0 ∧ b^2 - 4*a*c = 0) ↔ 
  (∃! x y : ℝ, a * x^2 + b * x + c = 0 ∧ y^2 = 4 * a * x)

/-- The main theorem: if the line 4x + 7y + k = 0 is tangent to the parabola y^2 = 16x, then k = 49 --/
theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4*x + 7*y + k = 0 → y^2 = 16*x) →
  (∃! x y : ℝ, 4*x + 7*y + k = 0 ∧ y^2 = 16*x) →
  k = 49 := by
  sorry


end NUMINAMATH_CALUDE_line_tangent_to_parabola_l382_38215


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l382_38220

theorem imaginary_unit_sum (i : ℂ) : i^2 = -1 → i + i^2 + i^3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l382_38220


namespace NUMINAMATH_CALUDE_base_number_problem_l382_38249

theorem base_number_problem (a : ℝ) : 
  (a > 0) → (a^14 - a^12 = 3 * a^12) → a = 2 := by sorry

end NUMINAMATH_CALUDE_base_number_problem_l382_38249


namespace NUMINAMATH_CALUDE_percentage_calculation_l382_38219

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 70 → (P / 100) * N - 10 = 25 → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l382_38219


namespace NUMINAMATH_CALUDE_x_interval_equivalence_l382_38224

theorem x_interval_equivalence (x : ℝ) : 
  (2/3 < x ∧ x < 3/4) ↔ (2 < 3*x ∧ 3*x < 3) ∧ (2 < 4*x ∧ 4*x < 3) := by
sorry

end NUMINAMATH_CALUDE_x_interval_equivalence_l382_38224


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l382_38235

structure Sibling where
  name : String
  pizza_fraction : ℚ

def pizza_problem (alex beth cyril eve dan : Sibling) : Prop :=
  alex.name = "Alex" ∧
  beth.name = "Beth" ∧
  cyril.name = "Cyril" ∧
  eve.name = "Eve" ∧
  dan.name = "Dan" ∧
  alex.pizza_fraction = 1/7 ∧
  beth.pizza_fraction = 1/5 ∧
  cyril.pizza_fraction = 1/6 ∧
  eve.pizza_fraction = 1/9 ∧
  dan.pizza_fraction = 1 - (alex.pizza_fraction + beth.pizza_fraction + cyril.pizza_fraction + eve.pizza_fraction)

theorem pizza_consumption_order (alex beth cyril eve dan : Sibling) 
  (h : pizza_problem alex beth cyril eve dan) :
  dan.pizza_fraction > beth.pizza_fraction ∧
  beth.pizza_fraction > cyril.pizza_fraction ∧
  cyril.pizza_fraction > alex.pizza_fraction ∧
  alex.pizza_fraction > eve.pizza_fraction :=
by sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l382_38235


namespace NUMINAMATH_CALUDE_star_equality_implies_x_equals_three_l382_38282

/-- Binary operation ⋆ on ordered pairs of integers -/
def star : (Int × Int) → (Int × Int) → (Int × Int) :=
  fun (a, b) (c, d) ↦ (a + c, b - d)

/-- Theorem stating that if (4,5) ⋆ (1,3) = (x,y) ⋆ (2,1), then x = 3 -/
theorem star_equality_implies_x_equals_three (x y : Int) :
  star (4, 5) (1, 3) = star (x, y) (2, 1) → x = 3 := by
  sorry


end NUMINAMATH_CALUDE_star_equality_implies_x_equals_three_l382_38282


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_minus_x_sq_range_of_a_for_nonempty_solution_l382_38226

-- Define the function f(x) = |x-1|
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part (1)
theorem solution_set_f_geq_1_minus_x_sq :
  {x : ℝ | f x ≥ 1 - x^2} = {x : ℝ | x ≤ 0 ∨ x ≥ 1} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_nonempty_solution (a : ℝ) :
  (∃ x : ℝ, f x < a - x^2 + |x + 1|) ↔ a > -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_minus_x_sq_range_of_a_for_nonempty_solution_l382_38226


namespace NUMINAMATH_CALUDE_minute_hand_catches_hour_hand_l382_38287

/-- The speed of the hour hand in degrees per minute -/
def hour_hand_speed : ℚ := 1/2

/-- The speed of the minute hand in degrees per minute -/
def minute_hand_speed : ℚ := 6

/-- The number of degrees in a full circle -/
def full_circle : ℚ := 360

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time after 12:00 when the minute hand first catches up to the hour hand -/
def catch_up_time : ℚ := 65 + 5/11

theorem minute_hand_catches_hour_hand :
  let relative_speed := minute_hand_speed - hour_hand_speed
  let catch_up_angle := catch_up_time * relative_speed
  catch_up_angle = full_circle ∧ 
  catch_up_time < minutes_per_hour := by
  sorry

#check minute_hand_catches_hour_hand

end NUMINAMATH_CALUDE_minute_hand_catches_hour_hand_l382_38287


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l382_38285

theorem no_simultaneous_squares : ¬∃ (m n : ℕ), ∃ (k l : ℕ), m^2 + n = k^2 ∧ n^2 + m = l^2 := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l382_38285


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l382_38280

/-- The number of people sharing the box of chocolate bars -/
def num_people : ℕ := 3

/-- The number of bars two people got combined -/
def bars_two_people : ℕ := 8

/-- The total number of bars in the box -/
def total_bars : ℕ := 16

/-- Theorem stating that the total number of bars is 16 -/
theorem chocolate_bar_count :
  (num_people : ℕ) = 3 →
  (bars_two_people : ℕ) = 8 →
  (total_bars : ℕ) = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_count_l382_38280


namespace NUMINAMATH_CALUDE_rectangle_area_inequality_l382_38295

theorem rectangle_area_inequality : ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 16 * 10 = 23 * 7 + ε := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_inequality_l382_38295


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_345_triangle_l382_38289

/-- The radius of the inscribed circle of a triangle with sides 3, 4, and 5 is 1 -/
theorem inscribed_circle_radius_345_triangle : 
  ∀ (a b c : ℝ) (r : ℝ), 
    a = 3 ∧ b = 4 ∧ c = 5 →
    (a + b + c) / 2 = 6 →
    r = 6 / ((a + b + c) / 2) →
    r = 1 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_345_triangle_l382_38289


namespace NUMINAMATH_CALUDE_angle_coincidence_l382_38210

def is_obtuse_angle (α : ℝ) : Prop := 90 < α ∧ α < 180

theorem angle_coincidence (α : ℝ) 
  (h1 : is_obtuse_angle α) 
  (h2 : (4 * α) % 360 = α % 360) : 
  α = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_coincidence_l382_38210


namespace NUMINAMATH_CALUDE_accumulator_implies_limit_in_segment_l382_38286

/-- A sequence is a function from natural numbers to real numbers -/
def Sequence := ℕ → ℝ

/-- A segment [a, b] is an accumulator for a sequence if infinitely many terms of the sequence lie within [a, b] -/
def IsAccumulator (s : Sequence) (a b : ℝ) : Prop :=
  ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧ a ≤ s n ∧ s n ≤ b

/-- The limit of a sequence, if it exists -/
def HasLimit (s : Sequence) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |s n - L| < ε

theorem accumulator_implies_limit_in_segment (s : Sequence) (a b L : ℝ) :
  IsAccumulator s a b → HasLimit s L → a ≤ L ∧ L ≤ b :=
by sorry


end NUMINAMATH_CALUDE_accumulator_implies_limit_in_segment_l382_38286


namespace NUMINAMATH_CALUDE_equal_sharing_contribution_l382_38278

def earnings : List ℕ := [10, 30, 50, 40, 70]

theorem equal_sharing_contribution :
  let total := earnings.sum
  let equal_share := total / earnings.length
  let max_earner := earnings.maximum?
  match max_earner with
  | some max => max - equal_share = 30
  | none => False
  := by sorry

end NUMINAMATH_CALUDE_equal_sharing_contribution_l382_38278


namespace NUMINAMATH_CALUDE_three_positions_from_six_people_l382_38259

/-- The number of ways to choose three distinct positions from a group of people -/
def choose_three_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- The number of people in the group -/
def group_size : ℕ := 6

/-- Theorem: The number of ways to choose a President, Vice-President, and Secretary 
    from a group of 6 people, where all positions must be filled by different individuals, 
    is equal to 120. -/
theorem three_positions_from_six_people : 
  choose_three_positions group_size = 120 := by sorry

end NUMINAMATH_CALUDE_three_positions_from_six_people_l382_38259


namespace NUMINAMATH_CALUDE_students_in_both_activities_l382_38246

theorem students_in_both_activities (total : ℕ) (band : ℕ) (sports : ℕ) (either : ℕ) 
  (h1 : total = 320)
  (h2 : band = 85)
  (h3 : sports = 200)
  (h4 : either = 225) :
  band + sports - either = 60 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_activities_l382_38246


namespace NUMINAMATH_CALUDE_sin_cos_sum_20_10_l382_38291

theorem sin_cos_sum_20_10 : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_20_10_l382_38291


namespace NUMINAMATH_CALUDE_smallest_divisible_by_nine_l382_38279

/-- The smallest digit d such that 528,d46 is divisible by 9 -/
def smallest_digit : ℕ := 2

/-- A function that constructs the number 528,d46 given a digit d -/
def construct_number (d : ℕ) : ℕ := 528000 + d * 100 + 46

theorem smallest_divisible_by_nine :
  (∀ d : ℕ, d < smallest_digit → ¬(9 ∣ construct_number d)) ∧
  (9 ∣ construct_number smallest_digit) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_nine_l382_38279


namespace NUMINAMATH_CALUDE_vector_properties_l382_38254

/-- Given two vectors a and b in ℝ³, prove properties about their components --/
theorem vector_properties (a b : ℝ × ℝ × ℝ) :
  let x := a.2.2
  let y := b.2.1
  (a = (2, 4, x) ∧ ‖a‖ = 6) →
  (x = 4 ∨ x = -4) ∧
  (a = (2, 4, x) ∧ b = (2, y, 2) ∧ ∃ (k : ℝ), a = k • b) →
  x + y = 6 := by sorry

end NUMINAMATH_CALUDE_vector_properties_l382_38254


namespace NUMINAMATH_CALUDE_greatest_n_value_l382_38218

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 12100) : n ≤ 10 ∧ ∃ m : ℤ, m = 10 ∧ 101 * m^2 ≤ 12100 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_l382_38218


namespace NUMINAMATH_CALUDE_sixth_graders_percentage_l382_38273

theorem sixth_graders_percentage (seventh_graders : ℕ) (seventh_graders_percentage : ℚ) (sixth_graders : ℕ) :
  seventh_graders = 64 →
  seventh_graders_percentage = 32 / 100 →
  sixth_graders = 76 →
  (sixth_graders : ℚ) / ((seventh_graders : ℚ) / seventh_graders_percentage) = 38 / 100 := by
  sorry

end NUMINAMATH_CALUDE_sixth_graders_percentage_l382_38273


namespace NUMINAMATH_CALUDE_all_statements_false_l382_38214

theorem all_statements_false :
  (∀ x : ℝ, x^2 = 4 → x = 2 ∨ x = -2) ∧
  (∀ x : ℝ, x^2 = 9 → x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, x^3 = -125 → x = -5) ∧
  (∀ x : ℝ, x^2 = 16 → x = 4 ∨ x = -4) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_false_l382_38214


namespace NUMINAMATH_CALUDE_fraction_modification_l382_38299

theorem fraction_modification (a b c d k : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : d ≠ 0) (h4 : k ≠ 0) (h5 : k ≠ 1) :
  let x := (b * c - a * d) / (k * d - c)
  (a + k * x) / (b + x) = c / d :=
by sorry

end NUMINAMATH_CALUDE_fraction_modification_l382_38299


namespace NUMINAMATH_CALUDE_shaded_area_regular_octagon_l382_38217

/-- The area of the shaded region in a regular octagon with side length 12 cm, 
    formed by connecting every other vertex (creating two squares) -/
theorem shaded_area_regular_octagon (side_length : ℝ) (h : side_length = 12) : 
  let octagon_area := 8 * (1/2 * side_length * (side_length / 2))
  octagon_area = 288 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_regular_octagon_l382_38217


namespace NUMINAMATH_CALUDE_a_gt_2_sufficient_not_necessary_for_a_sq_gt_2a_l382_38277

theorem a_gt_2_sufficient_not_necessary_for_a_sq_gt_2a :
  (∀ a : ℝ, a > 2 → a^2 > 2*a) ∧
  (∃ a : ℝ, a^2 > 2*a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_2_sufficient_not_necessary_for_a_sq_gt_2a_l382_38277


namespace NUMINAMATH_CALUDE_always_odd_l382_38237

theorem always_odd (n : ℤ) : ∃ k : ℤ, 2017 + 2*n = 2*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l382_38237


namespace NUMINAMATH_CALUDE_carina_coffee_amount_l382_38251

/-- The number of 10-ounce packages of coffee -/
def num_10oz_packages : ℕ := 4

/-- The number of 5-ounce packages of coffee -/
def num_5oz_packages : ℕ := num_10oz_packages + 2

/-- The total amount of coffee in ounces -/
def total_coffee : ℕ := num_10oz_packages * 10 + num_5oz_packages * 5

theorem carina_coffee_amount : total_coffee = 70 := by
  sorry

end NUMINAMATH_CALUDE_carina_coffee_amount_l382_38251


namespace NUMINAMATH_CALUDE_cube_surface_area_l382_38228

/-- Given a cube with volume 1728 cubic centimeters, its surface area is 864 square centimeters. -/
theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1728 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 864 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l382_38228


namespace NUMINAMATH_CALUDE_parallelogram_area_l382_38253

theorem parallelogram_area (base : ℝ) (slant_height : ℝ) (angle : ℝ) :
  base = 10 →
  slant_height = 6 →
  angle = 30 * π / 180 →
  base * (slant_height * Real.sin angle) = 30 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l382_38253


namespace NUMINAMATH_CALUDE_polynomial_property_l382_38213

-- Define the polynomial Q(x)
def Q (x a b c : ℝ) : ℝ := 3 * x^3 + a * x^2 + b * x + c

-- State the theorem
theorem polynomial_property (a b c : ℝ) :
  -- The y-intercept is 6
  Q 0 a b c = 6 →
  -- The mean of zeros, product of zeros, and sum of coefficients are equal
  (∃ m : ℝ, 
    -- Mean of zeros
    (-(a / 3) / 3 = m) ∧ 
    -- Product of zeros
    (-c / 3 = m) ∧ 
    -- Sum of coefficients
    (3 + a + b + c = m)) →
  -- Conclusion: b = -29
  b = -29 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l382_38213


namespace NUMINAMATH_CALUDE_customers_in_us_l382_38257

theorem customers_in_us (total : ℕ) (other_countries : ℕ) (h1 : total = 7422) (h2 : other_countries = 6699) :
  total - other_countries = 723 := by
  sorry

end NUMINAMATH_CALUDE_customers_in_us_l382_38257


namespace NUMINAMATH_CALUDE_jessica_rearrangement_time_l382_38230

/-- The time in hours required to write all rearrangements of a name -/
def time_to_write_rearrangements (name_length : ℕ) (repeated_letter_count : ℕ) (rearrangements_per_minute : ℕ) : ℚ :=
  let total_permutations := (name_length.factorial / repeated_letter_count.factorial : ℚ)
  let time_in_minutes := total_permutations / rearrangements_per_minute
  time_in_minutes / 60

/-- Theorem stating the time required to write all rearrangements of Jessica's name -/
theorem jessica_rearrangement_time :
  time_to_write_rearrangements 7 2 18 = 2333 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_jessica_rearrangement_time_l382_38230


namespace NUMINAMATH_CALUDE_elective_schemes_count_l382_38256

def total_courses : ℕ := 10
def courses_to_choose : ℕ := 3
def conflicting_courses : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem elective_schemes_count :
  (choose conflicting_courses 1 * choose (total_courses - conflicting_courses) (courses_to_choose - 1)) +
  (choose (total_courses - conflicting_courses) courses_to_choose) = 98 :=
by sorry

end NUMINAMATH_CALUDE_elective_schemes_count_l382_38256


namespace NUMINAMATH_CALUDE_incorrect_addition_statement_l382_38242

theorem incorrect_addition_statement : 
  (8 + 34 ≠ 32) ∧ (17 + 17 = 34) ∧ (15 + 13 = 28) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_addition_statement_l382_38242


namespace NUMINAMATH_CALUDE_choir_average_age_l382_38264

theorem choir_average_age 
  (num_females : ℕ) (num_males : ℕ) (num_children : ℕ)
  (avg_age_females : ℚ) (avg_age_males : ℚ) (avg_age_children : ℚ)
  (h1 : num_females = 12)
  (h2 : num_males = 20)
  (h3 : num_children = 8)
  (h4 : avg_age_females = 28)
  (h5 : avg_age_males = 38)
  (h6 : avg_age_children = 10) :
  (num_females * avg_age_females + num_males * avg_age_males + num_children * avg_age_children) / 
  (num_females + num_males + num_children : ℚ) = 1176 / 40 := by
  sorry

end NUMINAMATH_CALUDE_choir_average_age_l382_38264


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l382_38243

/-- A rhombus with a diagonal of length 6 and side length satisfying x^2 - 7x + 12 = 0 has a perimeter of 16 -/
theorem rhombus_perimeter (a b c d : ℝ) (h1 : a = b ∧ b = c ∧ c = d) 
  (h2 : ∃ (diag : ℝ), diag = 6) 
  (h3 : a^2 - 7*a + 12 = 0) : 
  a + b + c + d = 16 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l382_38243


namespace NUMINAMATH_CALUDE_zero_derivative_not_always_extremum_l382_38241

/-- A function f: ℝ → ℝ is differentiable -/
def DifferentiableFunction (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f

/-- x₀ is an extremum point of f if it's either a local maximum or minimum -/
def IsExtremumPoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  IsLocalMax f x₀ ∨ IsLocalMin f x₀

/-- The statement that if f'(x₀) = 0, then x₀ is an extremum point of f -/
def ZeroDerivativeImpliesExtremum (f : ℝ → ℝ) : Prop :=
  ∀ x₀ : ℝ, DifferentiableAt ℝ f x₀ → deriv f x₀ = 0 → IsExtremumPoint f x₀

theorem zero_derivative_not_always_extremum :
  ¬ (∀ f : ℝ → ℝ, DifferentiableFunction f → ZeroDerivativeImpliesExtremum f) :=
by sorry

end NUMINAMATH_CALUDE_zero_derivative_not_always_extremum_l382_38241


namespace NUMINAMATH_CALUDE_BI_length_is_15_over_4_l382_38261

/-- Two squares ABCD and EFGH with parallel sides -/
structure ParallelSquares :=
  (A B C D E F G H : ℝ × ℝ)

/-- Point where CG intersects BD -/
def I (squares : ParallelSquares) : ℝ × ℝ := sorry

/-- Length of BD -/
def BD_length (squares : ParallelSquares) : ℝ := 10

/-- Area of triangle BFC -/
def area_BFC (squares : ParallelSquares) : ℝ := 3

/-- Area of triangle CHD -/
def area_CHD (squares : ParallelSquares) : ℝ := 5

/-- Length of BI -/
def BI_length (squares : ParallelSquares) : ℝ := sorry

theorem BI_length_is_15_over_4 (squares : ParallelSquares) :
  BI_length squares = 15 / 4 := by sorry

end NUMINAMATH_CALUDE_BI_length_is_15_over_4_l382_38261


namespace NUMINAMATH_CALUDE_wise_men_hat_guesses_l382_38206

/-- Represents the maximum number of guaranteed correct hat color guesses -/
def max_guaranteed_correct_guesses (n k : ℕ) : ℕ :=
  n - k - 1

/-- Theorem stating the maximum number of guaranteed correct hat color guesses -/
theorem wise_men_hat_guesses (n k : ℕ) (h1 : k < n) :
  max_guaranteed_correct_guesses n k = n - k - 1 :=
by sorry

end NUMINAMATH_CALUDE_wise_men_hat_guesses_l382_38206


namespace NUMINAMATH_CALUDE_max_value_inequality_l382_38238

theorem max_value_inequality (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  (x * y * z * w * (x + y + z + w)) / ((x + y)^3 * (y + z + w)^2) ≤ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l382_38238


namespace NUMINAMATH_CALUDE_ceiling_minus_x_value_l382_38272

theorem ceiling_minus_x_value (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ δ : ℝ, 0 < δ ∧ δ < 1 ∧ ⌈x⌉ - x = 1 - δ := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_value_l382_38272


namespace NUMINAMATH_CALUDE_find_divisor_l382_38205

theorem find_divisor (d : ℕ) (h1 : d > 0) (h2 : 1050 % d = 0) (h3 : 1049 % d ≠ 0) : d = 1050 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l382_38205


namespace NUMINAMATH_CALUDE_tangent_lines_slope_4_tangent_line_at_point_2_neg6_l382_38211

-- Define the function f(x) = x³ + x - 16
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_lines_slope_4 (x y : ℝ) :
  (4 * x - y - 18 = 0 ∨ 4 * x - y - 14 = 0) →
  ∃ x₀ : ℝ, f' x₀ = 4 ∧ y = f x₀ + 4 * (x - x₀) :=
sorry

theorem tangent_line_at_point_2_neg6 (x y : ℝ) :
  13 * x - y - 32 = 0 →
  y = f 2 + f' 2 * (x - 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_slope_4_tangent_line_at_point_2_neg6_l382_38211


namespace NUMINAMATH_CALUDE_solve_system_for_x_l382_38248

theorem solve_system_for_x :
  ∀ (x y : ℚ),
  (3 * x - 2 * y = 8) →
  (x + 3 * y = 7) →
  x = 38 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_for_x_l382_38248


namespace NUMINAMATH_CALUDE_solutions_for_15_l382_38208

/-- The number of different integer solutions for |x| + |y| = n -/
def numSolutions (n : ℕ) : ℕ :=
  4 * n

theorem solutions_for_15 : numSolutions 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_solutions_for_15_l382_38208


namespace NUMINAMATH_CALUDE_min_value_quadratic_l382_38255

theorem min_value_quadratic (x : ℝ) : 
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 5*x^2 + 20*x + 45 → y ≥ y_min ∧ y_min = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l382_38255


namespace NUMINAMATH_CALUDE_wire_length_ratio_l382_38298

/-- The length of one piece of wire used in Bonnie's cube frame -/
def bonnie_wire_length : ℝ := 8

/-- The number of wire pieces used in Bonnie's cube frame -/
def bonnie_wire_count : ℕ := 12

/-- The length of one piece of wire used in Roark's unit cube frames -/
def roark_wire_length : ℝ := 2

/-- The volume of Bonnie's cube -/
def bonnie_cube_volume : ℝ := bonnie_wire_length ^ 3

/-- The volume of one of Roark's unit cubes -/
def roark_unit_cube_volume : ℝ := roark_wire_length ^ 3

/-- The number of wire pieces needed for one of Roark's unit cube frames -/
def roark_wire_count_per_cube : ℕ := 12

theorem wire_length_ratio :
  (bonnie_wire_count * bonnie_wire_length) / 
  (((bonnie_cube_volume / roark_unit_cube_volume) : ℝ) * 
   (roark_wire_count_per_cube : ℝ) * roark_wire_length) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l382_38298


namespace NUMINAMATH_CALUDE_sam_bought_cards_l382_38222

/-- The number of baseball cards Sam bought from Mike -/
def cards_bought (initial_cards current_cards : ℕ) : ℕ :=
  initial_cards - current_cards

/-- Theorem stating that the number of cards Sam bought is the difference between Mike's initial and current number of cards -/
theorem sam_bought_cards (mike_initial mike_current : ℕ) 
  (h1 : mike_initial = 87) 
  (h2 : mike_current = 74) : 
  cards_bought mike_initial mike_current = 13 := by
  sorry

end NUMINAMATH_CALUDE_sam_bought_cards_l382_38222


namespace NUMINAMATH_CALUDE_probability_N18_mod7_equals_1_is_2_7_l382_38268

/-- The probability that N^18 mod 7 = 1, given N is an odd integer randomly chosen from 1 to 2023 -/
def probability_N18_mod7_equals_1 : ℚ :=
  let N := Finset.filter (fun n => n % 2 = 1) (Finset.range 2023)
  let favorable := N.filter (fun n => (n^18) % 7 = 1)
  favorable.card / N.card

theorem probability_N18_mod7_equals_1_is_2_7 :
  probability_N18_mod7_equals_1 = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_probability_N18_mod7_equals_1_is_2_7_l382_38268


namespace NUMINAMATH_CALUDE_fraction_inequality_l382_38258

theorem fraction_inequality (x : ℝ) (h : x ≠ 1) :
  (1 / (x - 1) ≤ 1) ↔ (x < 1 ∨ x ≥ 2) := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l382_38258


namespace NUMINAMATH_CALUDE_dianna_problem_l382_38265

def correct_expression (f : ℤ) : ℤ := 1 - (2 - (3 - (4 + (5 - f))))

def misinterpreted_expression (f : ℤ) : ℤ := 1 - 2 - 3 - 4 + 5 - f

theorem dianna_problem : ∃ f : ℤ, correct_expression f = misinterpreted_expression f ∧ f = 2 := by
  sorry

end NUMINAMATH_CALUDE_dianna_problem_l382_38265


namespace NUMINAMATH_CALUDE_sin_30_tan_45_calculation_l382_38274

theorem sin_30_tan_45_calculation : 2 * Real.sin (30 * π / 180) - Real.tan (45 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_tan_45_calculation_l382_38274


namespace NUMINAMATH_CALUDE_rower_upstream_speed_l382_38271

/-- Calculates the upstream speed of a rower given their still water speed and downstream speed. -/
def upstream_speed (still_water_speed downstream_speed : ℝ) : ℝ :=
  2 * still_water_speed - downstream_speed

/-- Proves that given a man's speed in still water is 35 kmph and his downstream speed is 45 kmph, his upstream speed is 25 kmph. -/
theorem rower_upstream_speed :
  let still_water_speed := (35 : ℝ)
  let downstream_speed := (45 : ℝ)
  upstream_speed still_water_speed downstream_speed = 25 := by
sorry

#eval upstream_speed 35 45

end NUMINAMATH_CALUDE_rower_upstream_speed_l382_38271


namespace NUMINAMATH_CALUDE_exterior_angle_of_right_triangle_l382_38245

-- Define a triangle
structure Triangle :=
  (A B C : ℝ)
  (sum_of_angles : A + B + C = 180)

-- Define a right triangle
structure RightTriangle extends Triangle :=
  (right_angle : C = 90)

-- Theorem statement
theorem exterior_angle_of_right_triangle (t : RightTriangle) :
  180 - t.C = 90 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_of_right_triangle_l382_38245


namespace NUMINAMATH_CALUDE_keith_initial_pears_keith_initial_pears_proof_l382_38263

/-- Proves that Keith picked 47 pears initially -/
theorem keith_initial_pears : ℕ :=
  let mike_pears : ℕ := 12
  let keith_gave_away : ℕ := 46
  let remaining_pears : ℕ := 13
  47

theorem keith_initial_pears_proof (mike_pears keith_gave_away remaining_pears : ℕ) 
  (h1 : mike_pears = 12)
  (h2 : keith_gave_away = 46)
  (h3 : remaining_pears = 13)
  (h4 : keith_initial_pears - keith_gave_away + mike_pears = remaining_pears) :
  keith_initial_pears = 47 := by
  sorry

end NUMINAMATH_CALUDE_keith_initial_pears_keith_initial_pears_proof_l382_38263


namespace NUMINAMATH_CALUDE_hard_candy_colouring_amount_l382_38212

/-- Represents the candy store's daily production and food colouring usage --/
structure CandyStore where
  lollipop_colouring : ℕ  -- ml of food colouring per lollipop
  lollipops_made : ℕ      -- number of lollipops made
  hard_candies_made : ℕ   -- number of hard candies made
  total_colouring : ℕ     -- total ml of food colouring used

/-- Calculates the amount of food colouring needed for each hard candy --/
def hard_candy_colouring (store : CandyStore) : ℕ :=
  (store.total_colouring - store.lollipop_colouring * store.lollipops_made) / store.hard_candies_made

/-- Theorem stating the amount of food colouring needed for each hard candy --/
theorem hard_candy_colouring_amount (store : CandyStore)
  (h1 : store.lollipop_colouring = 5)
  (h2 : store.lollipops_made = 100)
  (h3 : store.hard_candies_made = 5)
  (h4 : store.total_colouring = 600) :
  hard_candy_colouring store = 20 := by
  sorry

end NUMINAMATH_CALUDE_hard_candy_colouring_amount_l382_38212


namespace NUMINAMATH_CALUDE_team_a_more_uniform_heights_l382_38252

/-- Represents a team with its height statistics -/
structure Team where
  averageHeight : ℝ
  variance : ℝ

/-- Defines when a team has more uniform heights than another -/
def hasMoreUniformHeights (t1 t2 : Team) : Prop :=
  t1.variance < t2.variance

/-- Theorem stating that Team A has more uniform heights than Team B -/
theorem team_a_more_uniform_heights :
  let teamA : Team := { averageHeight := 1.82, variance := 0.56 }
  let teamB : Team := { averageHeight := 1.82, variance := 2.1 }
  hasMoreUniformHeights teamA teamB := by
  sorry

end NUMINAMATH_CALUDE_team_a_more_uniform_heights_l382_38252


namespace NUMINAMATH_CALUDE_complex_coordinate_i_times_2_minus_i_l382_38225

theorem complex_coordinate_i_times_2_minus_i : 
  (Complex.I * (2 - Complex.I)).re = 1 ∧ (Complex.I * (2 - Complex.I)).im = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinate_i_times_2_minus_i_l382_38225


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l382_38240

theorem unique_positive_integer_solution : ∃! (x : ℕ), x > 0 ∧ (4 * x)^2 - 3 * x = 1764 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l382_38240


namespace NUMINAMATH_CALUDE_expression_value_l382_38294

theorem expression_value (a b m n x : ℝ) 
  (h1 : a = -b) 
  (h2 : m * n = 1) 
  (h3 : |x| = 2) : 
  -2*m*n + 3*(a+b) - x = -4 ∨ -2*m*n + 3*(a+b) - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l382_38294


namespace NUMINAMATH_CALUDE_derivative_of_square_root_l382_38234

theorem derivative_of_square_root (x : ℝ) (h : x > 0) :
  deriv (fun x => Real.sqrt x) x = 1 / (2 * Real.sqrt x) := by
sorry

end NUMINAMATH_CALUDE_derivative_of_square_root_l382_38234


namespace NUMINAMATH_CALUDE_range_of_function_l382_38204

theorem range_of_function : 
  ∀ (x : ℝ), 12 ≤ |x + 5| - |x - 3| + 4 ∧ 
  (∃ (x₁ x₂ : ℝ), |x₁ + 5| - |x₁ - 3| + 4 = 12 ∧ |x₂ + 5| - |x₂ - 3| + 4 = 18) ∧
  (∀ (y : ℝ), (∃ (x : ℝ), y = |x + 5| - |x - 3| + 4) → 12 ≤ y ∧ y ≤ 18) :=
by sorry

end NUMINAMATH_CALUDE_range_of_function_l382_38204


namespace NUMINAMATH_CALUDE_jose_play_time_l382_38260

/-- Calculates the total hours played given the time spent on football and basketball in minutes -/
def total_hours_played (football_minutes : ℕ) (basketball_minutes : ℕ) : ℚ :=
  (football_minutes + basketball_minutes : ℚ) / 60

/-- Theorem stating that playing football for 30 minutes and basketball for 60 minutes results in 1.5 hours of total play time -/
theorem jose_play_time : total_hours_played 30 60 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_jose_play_time_l382_38260


namespace NUMINAMATH_CALUDE_union_complement_theorem_l382_38209

def I : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {2}

theorem union_complement_theorem :
  B ∪ (I \ A) = {2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_union_complement_theorem_l382_38209


namespace NUMINAMATH_CALUDE_ball_count_theorem_l382_38262

theorem ball_count_theorem (red_balls : ℕ) (white_balls : ℕ) (total_balls : ℕ) (prob_red : ℚ) : 
  red_balls = 4 →
  total_balls = red_balls + white_balls →
  prob_red = 1/4 →
  (red_balls : ℚ) / total_balls = prob_red →
  white_balls = 12 := by
sorry

end NUMINAMATH_CALUDE_ball_count_theorem_l382_38262


namespace NUMINAMATH_CALUDE_expression_equality_l382_38236

theorem expression_equality : 
  |Real.sqrt 3 - 3| - Real.sqrt 16 + Real.cos (30 * π / 180) + (1/3)^0 = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l382_38236
