import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_sum_independence_l632_63238

theorem polynomial_sum_independence (a b : ℝ) :
  (∀ x y : ℝ, (x^2 + a*x - y + b) + (b*x^2 - 3*x + 6*y - 3) = (5*y + b - 3)) →
  3*(a^2 - 2*a*b + b^2) - (4*a^2 - 2*(1/2*a^2 + a*b - 3/2*b^2)) = 12 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_independence_l632_63238


namespace NUMINAMATH_CALUDE_highway_vehicles_l632_63232

/-- The number of vehicles involved in accidents per 100 million vehicles -/
def accident_rate : ℕ := 40

/-- The total number of vehicles involved in accidents last year -/
def total_accidents : ℕ := 800

/-- The number of vehicles per 100 million for the accident rate calculation -/
def base_vehicles : ℕ := 100000000

/-- The number of vehicles that traveled on the highway last year -/
def total_vehicles : ℕ := 2000000000

theorem highway_vehicles :
  total_vehicles = (total_accidents * base_vehicles) / accident_rate :=
sorry

end NUMINAMATH_CALUDE_highway_vehicles_l632_63232


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l632_63267

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a/c = b/d = 4/5, then the ratio of their areas is 16/25. -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 4 / 5) (h2 : b / d = 4 / 5) :
  (a * b) / (c * d) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l632_63267


namespace NUMINAMATH_CALUDE_ratio_cubes_equals_729_l632_63234

theorem ratio_cubes_equals_729 : (81000 ^ 3) / (9000 ^ 3) = 729 := by
  sorry

end NUMINAMATH_CALUDE_ratio_cubes_equals_729_l632_63234


namespace NUMINAMATH_CALUDE_min_value_I_l632_63272

theorem min_value_I (a b c x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x ≥ 0) (hy : y ≥ 0)
  (h_sum : a^6 + b^6 + c^6 = 3)
  (h_constraint : (x + 1)^2 + y^2 ≤ 2) : 
  let I := 1 / (2*a^3*x + b^3*y^2) + 1 / (2*b^3*x + c^3*y^2) + 1 / (2*c^3*x + a^3*y^2)
  ∀ I', I ≥ I' → I' ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_I_l632_63272


namespace NUMINAMATH_CALUDE_problem_solution_l632_63204

theorem problem_solution : 9 - (3 / (1 / 3) + 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l632_63204


namespace NUMINAMATH_CALUDE_unfactorable_quartic_l632_63207

theorem unfactorable_quartic : ¬∃ (a b c d : ℤ), ∀ (x : ℝ), 
  x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end NUMINAMATH_CALUDE_unfactorable_quartic_l632_63207


namespace NUMINAMATH_CALUDE_shea_corn_purchase_l632_63243

/-- The cost of corn per pound in cents -/
def corn_cost : ℕ := 110

/-- The cost of beans per pound in cents -/
def bean_cost : ℕ := 50

/-- The total number of pounds of corn and beans bought -/
def total_pounds : ℕ := 30

/-- The total cost in cents -/
def total_cost : ℕ := 2100

/-- The number of pounds of corn bought -/
def corn_pounds : ℚ := 10

theorem shea_corn_purchase :
  ∃ (bean_pounds : ℚ),
    bean_pounds + corn_pounds = total_pounds ∧
    bean_cost * bean_pounds + corn_cost * corn_pounds = total_cost :=
by sorry

end NUMINAMATH_CALUDE_shea_corn_purchase_l632_63243


namespace NUMINAMATH_CALUDE_work_completion_time_work_completion_result_l632_63251

/-- The time taken to complete a work when two people work together, given their individual completion times -/
theorem work_completion_time (ajay_time vijay_time : ℝ) (h1 : ajay_time > 0) (h2 : vijay_time > 0) :
  (ajay_time * vijay_time) / (ajay_time + vijay_time) = 
    (8 : ℝ) * 24 / ((8 : ℝ) + 24) :=
by sorry

/-- The result of the work completion time calculation is 6 days -/
theorem work_completion_result :
  (8 : ℝ) * 24 / ((8 : ℝ) + 24) = 6 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_work_completion_result_l632_63251


namespace NUMINAMATH_CALUDE_third_grade_students_l632_63293

theorem third_grade_students (total : ℕ) (male female : ℕ) : 
  total = 41 → 
  male = female + 3 → 
  total = male + female →
  male = 22 := by
sorry

end NUMINAMATH_CALUDE_third_grade_students_l632_63293


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l632_63224

theorem quadratic_roots_sum (a b : ℝ) : 
  (a^2 + 8*a + 4 = 0) → 
  (b^2 + 8*b + 4 = 0) → 
  (a ≠ 0) →
  (b ≠ 0) →
  (a/b + b/a = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l632_63224


namespace NUMINAMATH_CALUDE_optimal_plan_l632_63217

/-- Represents the unit price of type A prizes -/
def price_A : ℝ := 30

/-- Represents the unit price of type B prizes -/
def price_B : ℝ := 15

/-- The total number of prizes to purchase -/
def total_prizes : ℕ := 30

/-- Condition: Total cost of 3 type A and 2 type B prizes is 120 yuan -/
axiom condition1 : 3 * price_A + 2 * price_B = 120

/-- Condition: Total cost of 5 type A and 4 type B prizes is 210 yuan -/
axiom condition2 : 5 * price_A + 4 * price_B = 210

/-- Function to calculate the total cost given the number of type A prizes -/
def total_cost (num_A : ℕ) : ℝ :=
  price_A * num_A + price_B * (total_prizes - num_A)

/-- Theorem stating the most cost-effective plan and its total cost -/
theorem optimal_plan :
  ∃ (num_A : ℕ),
    num_A ≥ (total_prizes - num_A) / 3 ∧
    num_A = 8 ∧
    total_cost num_A = 570 ∧
    ∀ (other_num_A : ℕ),
      other_num_A ≥ (total_prizes - other_num_A) / 3 →
      total_cost other_num_A ≥ total_cost num_A :=
sorry

end NUMINAMATH_CALUDE_optimal_plan_l632_63217


namespace NUMINAMATH_CALUDE_solution_value_l632_63278

-- Define the equations
def equation1 (m x : ℝ) : Prop := (m + 3) * x^(|m| - 2) + 6 * m = 0
def equation2 (n x : ℝ) : Prop := n * x - 5 = x * (3 - n)

-- Define the linearity condition for equation1
def equation1_is_linear (m : ℝ) : Prop := |m| - 2 = 0

-- Define the main theorem
theorem solution_value (m n x : ℝ) :
  (∀ y : ℝ, equation1 m y ↔ equation2 n y) →
  equation1_is_linear m →
  (m + x)^2000 * (-m^2 * n + x * n^2) + 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_value_l632_63278


namespace NUMINAMATH_CALUDE_ball_arrangements_count_l632_63222

/-- The number of ways to arrange guests in circles with alternating hat colors -/
def ball_arrangements (N : ℕ) : ℕ := (2 * N).factorial

/-- Theorem stating that the number of valid arrangements is (2N)! -/
theorem ball_arrangements_count (N : ℕ) :
  ball_arrangements N = (2 * N).factorial :=
by sorry

end NUMINAMATH_CALUDE_ball_arrangements_count_l632_63222


namespace NUMINAMATH_CALUDE_pen_pencil_cost_ratio_l632_63255

/-- Given a pen and pencil with a total cost of $6, where the pen costs $4,
    prove that the ratio of the cost of the pen to the cost of the pencil is 4:1. -/
theorem pen_pencil_cost_ratio :
  ∀ (pen_cost pencil_cost : ℚ),
  pen_cost + pencil_cost = 6 →
  pen_cost = 4 →
  pen_cost / pencil_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_pen_pencil_cost_ratio_l632_63255


namespace NUMINAMATH_CALUDE_larger_segment_approx_59_l632_63263

/-- Triangle with sides 40, 50, and 110 units --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 40
  hb : b = 50
  hc : c = 110

/-- Altitude dropped on the longest side --/
def altitude (t : Triangle) : ℝ := sorry

/-- Larger segment cut off on the longest side --/
def larger_segment (t : Triangle) : ℝ := sorry

/-- Theorem stating that the larger segment is approximately 59 units --/
theorem larger_segment_approx_59 (t : Triangle) :
  |larger_segment t - 59| < 0.5 := by sorry

end NUMINAMATH_CALUDE_larger_segment_approx_59_l632_63263


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l632_63201

def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | -4 < x ∧ x < 1 ∨ 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l632_63201


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l632_63262

/-- The sum of the first n 9's (e.g., 9, 99, 999, ...) -/
def sumOfNines (n : ℕ) : ℕ := (10^n - 1)

/-- The sum of the digits of a natural number -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digitSum (n / 10)

/-- M is defined as the sum of the first five 9's -/
def M : ℕ := (sumOfNines 1) + (sumOfNines 2) + (sumOfNines 3) + (sumOfNines 4) + (sumOfNines 5)

theorem sum_of_digits_M : digitSum M = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l632_63262


namespace NUMINAMATH_CALUDE_factorization_of_expression_l632_63271

theorem factorization_of_expression (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (b^2 * c^3) := by sorry

end NUMINAMATH_CALUDE_factorization_of_expression_l632_63271


namespace NUMINAMATH_CALUDE_power_equation_solution_l632_63239

theorem power_equation_solution (n : ℕ) : 3^n = 3 * 9^3 * 81^2 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l632_63239


namespace NUMINAMATH_CALUDE_scenario_is_simple_random_sampling_l632_63252

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | ComplexRandom

/-- Represents a population of students -/
structure Population where
  size : ℕ
  is_first_year : Bool

/-- Represents a sample from a population -/
structure Sample where
  size : ℕ
  population : Population
  selection_method : SamplingMethod

/-- The sampling method used in the given scenario -/
def scenario_sampling : Sample where
  size := 20
  population := { size := 200, is_first_year := true }
  selection_method := SamplingMethod.SimpleRandom

/-- Theorem stating that the sampling method used in the scenario is simple random sampling -/
theorem scenario_is_simple_random_sampling :
  scenario_sampling.selection_method = SamplingMethod.SimpleRandom :=
by
  sorry


end NUMINAMATH_CALUDE_scenario_is_simple_random_sampling_l632_63252


namespace NUMINAMATH_CALUDE_matrix_equality_l632_63209

theorem matrix_equality (X Y : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : X + Y = X * Y)
  (h2 : X * Y = ![![16/3, 2], ![-10/3, 10/3]]) :
  Y * X = ![![16/3, 2], ![-10/3, 10/3]] := by sorry

end NUMINAMATH_CALUDE_matrix_equality_l632_63209


namespace NUMINAMATH_CALUDE_minimum_cost_for_boxes_l632_63203

/-- The dimensions of a box in inches -/
def box_dimensions : Fin 3 → ℕ
  | 0 => 20
  | 1 => 20
  | 2 => 15
  | _ => 0

/-- The volume of a single box in cubic inches -/
def box_volume : ℕ := (box_dimensions 0) * (box_dimensions 1) * (box_dimensions 2)

/-- The cost of a single box in cents -/
def box_cost : ℕ := 50

/-- The total volume of the collection in cubic inches -/
def collection_volume : ℕ := 3060000

/-- The number of boxes needed to package the collection -/
def boxes_needed : ℕ := (collection_volume + box_volume - 1) / box_volume

theorem minimum_cost_for_boxes : 
  boxes_needed * box_cost = 25500 :=
sorry

end NUMINAMATH_CALUDE_minimum_cost_for_boxes_l632_63203


namespace NUMINAMATH_CALUDE_polynomial_root_implies_k_value_l632_63291

theorem polynomial_root_implies_k_value : ∀ k : ℚ,
  (3 : ℚ)^3 + k * 3 + 20 = 0 → k = -47/3 := by sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_k_value_l632_63291


namespace NUMINAMATH_CALUDE_power_of_two_equality_l632_63282

theorem power_of_two_equality : (2^8)^5 = 2^8 * 2^32 := by sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l632_63282


namespace NUMINAMATH_CALUDE_power_product_evaluation_l632_63248

theorem power_product_evaluation : 
  let a : ℕ := 2
  (a^3 * a^4 : ℕ) = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_product_evaluation_l632_63248


namespace NUMINAMATH_CALUDE_equation_solution_l632_63294

theorem equation_solution (m n x : ℝ) (hm : m > 0) (hn : n < 0) :
  (x - m)^2 - (x - n)^2 = (m - n)^2 → x = m :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l632_63294


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l632_63229

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l632_63229


namespace NUMINAMATH_CALUDE_angle_terminal_side_range_l632_63285

theorem angle_terminal_side_range (θ : Real) (a : Real) :
  (∃ (x y : Real), x = a - 2 ∧ y = a + 2 ∧ x = y * Real.tan θ) →
  Real.cos θ ≤ 0 →
  Real.sin θ > 0 →
  a ∈ Set.Ioo (-2) 2 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_range_l632_63285


namespace NUMINAMATH_CALUDE_max_a_value_l632_63299

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - 3 > 0 → x < a) ∧ 
  (∃ x : ℝ, x < a ∧ x^2 - 2*x - 3 ≤ 0) →
  a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l632_63299


namespace NUMINAMATH_CALUDE_cubic_function_property_l632_63289

/-- Given a cubic function f(x) = ax³ - bx + 1 where a and b are real numbers,
    if f(-2) = -1, then f(2) = 3 -/
theorem cubic_function_property (a b : ℝ) :
  (fun x => a * x^3 - b * x + 1) (-2) = -1 →
  (fun x => a * x^3 - b * x + 1) 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l632_63289


namespace NUMINAMATH_CALUDE_distinct_cube_edge_colorings_l632_63202

/-- The group of rotations of the cube -/
structure CubeRotationGroup where
  D : Type
  mul : D → D → D

/-- The permutation group of the edges of the cube induced by the rotation group -/
structure EdgePermutationGroup where
  W : Type
  comp : W → W → W

/-- The cycle index polynomial for the permutation group (W, ∘) -/
def cycle_index_polynomial (W : EdgePermutationGroup) : ℕ :=
  sorry

/-- The number of distinct colorings for a given permutation type -/
def colorings_for_permutation (perm_type : String) : ℕ :=
  sorry

/-- Theorem: The number of distinct ways to color the edges of a cube with 3 red, 3 blue, and 6 yellow edges is 780 -/
theorem distinct_cube_edge_colorings :
  let num_edges : ℕ := 12
  let num_red : ℕ := 3
  let num_blue : ℕ := 3
  let num_yellow : ℕ := 6
  (num_red + num_blue + num_yellow = num_edges) →
  (∃ (W : EdgePermutationGroup),
    (cycle_index_polynomial W *
     (colorings_for_permutation "1^12" +
      8 * colorings_for_permutation "3^4" +
      6 * colorings_for_permutation "1^2 2^5")) / 24 = 780) :=
by
  sorry

end NUMINAMATH_CALUDE_distinct_cube_edge_colorings_l632_63202


namespace NUMINAMATH_CALUDE_average_difference_l632_63261

theorem average_difference (x : ℝ) : 
  (10 + 30 + 50) / 3 = ((20 + 40 + x) / 3) + 8 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l632_63261


namespace NUMINAMATH_CALUDE_twin_prime_power_sum_divisibility_l632_63276

theorem twin_prime_power_sum_divisibility (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → q = p + 2 → (p + q) ∣ (p^q + q^p) := by
  sorry

end NUMINAMATH_CALUDE_twin_prime_power_sum_divisibility_l632_63276


namespace NUMINAMATH_CALUDE_octagon_semicircles_area_l632_63212

/-- The area of the region inside a regular octagon with side length 3 and eight inscribed semicircles --/
theorem octagon_semicircles_area : 
  let s : Real := 3  -- side length of the octagon
  let r : Real := s / 2  -- radius of each semicircle
  let octagon_area : Real := 2 * (1 + Real.sqrt 2) * s^2
  let semicircle_area : Real := π * r^2 / 2
  let total_semicircle_area : Real := 8 * semicircle_area
  octagon_area - total_semicircle_area = 18 * (1 + Real.sqrt 2) - 9 * π := by
sorry

end NUMINAMATH_CALUDE_octagon_semicircles_area_l632_63212


namespace NUMINAMATH_CALUDE_amusement_park_ride_orders_l632_63268

theorem amusement_park_ride_orders : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_ride_orders_l632_63268


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l632_63241

/-- In a right triangle with one angle of 30° and the side opposite to this angle
    having length 6, the length of the hypotenuse is 12. -/
theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  a = 6 →  -- Length of the side opposite to 30° angle
  Real.cos (30 * π / 180) = b / c →  -- Cosine of 30° in terms of adjacent side and hypotenuse
  c = 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l632_63241


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l632_63281

theorem trigonometric_expression_equality :
  let sin30 := (1 : ℝ) / 2
  let cos30 := Real.sqrt 3 / 2
  let tan60 := Real.sqrt 3
  2 * sin30 + cos30 * tan60 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l632_63281


namespace NUMINAMATH_CALUDE_enjoyable_gameplay_l632_63275

theorem enjoyable_gameplay (total_hours : ℝ) (boring_percentage : ℝ) (expansion_hours : ℝ) :
  total_hours = 100 ∧ 
  boring_percentage = 80 ∧ 
  expansion_hours = 30 →
  (1 - boring_percentage / 100) * total_hours + expansion_hours = 50 := by
  sorry

end NUMINAMATH_CALUDE_enjoyable_gameplay_l632_63275


namespace NUMINAMATH_CALUDE_min_distance_to_line_l632_63277

/-- Given a right triangle with sides a, b, and hypotenuse c, and a point (m, n) on the line ax + by + 2c = 0, 
    the minimum value of m^2 + n^2 is 4. -/
theorem min_distance_to_line (a b c m n : ℝ) : 
  a^2 + b^2 = c^2 →  -- Right triangle condition
  a * m + b * n + 2 * c = 0 →  -- Point (m, n) lies on the line
  ∃ (m₀ n₀ : ℝ), a * m₀ + b * n₀ + 2 * c = 0 ∧ 
    ∀ (m' n' : ℝ), a * m' + b * n' + 2 * c = 0 → m₀^2 + n₀^2 ≤ m'^2 + n'^2 ∧
    m₀^2 + n₀^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l632_63277


namespace NUMINAMATH_CALUDE_no_special_sequence_exists_l632_63296

theorem no_special_sequence_exists : ¬ ∃ (a : ℕ → ℕ),
  (∀ n : ℕ, a n < a (n + 1)) ∧
  (∃ N : ℕ, ∀ m : ℕ, m ≥ N →
    ∃! (i j : ℕ), m = a i + a j) :=
by sorry

end NUMINAMATH_CALUDE_no_special_sequence_exists_l632_63296


namespace NUMINAMATH_CALUDE_max_value_of_vector_sum_l632_63208

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem max_value_of_vector_sum (a b c : V) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (hc : ‖c‖ = 3) :
  ∃ (max_value : ℝ), max_value = 94 ∧
    ∀ (x y z : V), ‖x‖ = 1 → ‖y‖ = 2 → ‖z‖ = 3 →
      ‖x + 2•y‖^2 + ‖y + 2•z‖^2 + ‖z + 2•x‖^2 ≤ max_value :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_vector_sum_l632_63208


namespace NUMINAMATH_CALUDE_percentage_problem_l632_63292

theorem percentage_problem (P : ℝ) : 
  (5 / 100 * 6400 = P / 100 * 650 + 190) → P = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l632_63292


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l632_63288

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s ^ 2 = 64 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l632_63288


namespace NUMINAMATH_CALUDE_total_study_hours_l632_63253

/-- The number of weeks in the fall semester -/
def semester_weeks : ℕ := 15

/-- The number of study hours on weekdays -/
def weekday_hours : ℕ := 3

/-- The number of study hours on Saturday -/
def saturday_hours : ℕ := 4

/-- The number of study hours on Sunday -/
def sunday_hours : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- Theorem stating the total study hours during the semester -/
theorem total_study_hours :
  semester_weeks * (weekdays_per_week * weekday_hours + saturday_hours + sunday_hours) = 360 := by
  sorry

end NUMINAMATH_CALUDE_total_study_hours_l632_63253


namespace NUMINAMATH_CALUDE_boris_climbs_needed_l632_63236

def hugo_elevation : ℕ := 10000
def boris_elevation : ℕ := hugo_elevation - 2500
def hugo_climbs : ℕ := 3

theorem boris_climbs_needed : 
  (hugo_elevation * hugo_climbs) / boris_elevation = 4 := by sorry

end NUMINAMATH_CALUDE_boris_climbs_needed_l632_63236


namespace NUMINAMATH_CALUDE_determinant_equals_t_minus_s_plus_r_l632_63230

-- Define the polynomial
def polynomial (x r s t : ℝ) : ℝ := x^4 + r*x^2 + s*x + t

-- Define the matrix
def matrix (a b c d : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![1+a, 1,   1,   1],
    ![1,   1+b, 1,   1],
    ![1,   1,   1+c, 1],
    ![1,   1,   1,   1+d]]

theorem determinant_equals_t_minus_s_plus_r 
  (r s t : ℝ) (a b c d : ℝ) 
  (h1 : polynomial a r s t = 0)
  (h2 : polynomial b r s t = 0)
  (h3 : polynomial c r s t = 0)
  (h4 : polynomial d r s t = 0) :
  Matrix.det (matrix a b c d) = t - s + r := by
  sorry

end NUMINAMATH_CALUDE_determinant_equals_t_minus_s_plus_r_l632_63230


namespace NUMINAMATH_CALUDE_smallest_x_value_l632_63274

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (254 + x)) : 
  x ≥ 2 ∧ ∃ (y' : ℕ+), (3 : ℚ) / 4 = y' / (254 + 2) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l632_63274


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l632_63297

def first_six_primes : List ℕ := [2, 3, 5, 7, 11, 13]
def seventh_prime : ℕ := 17

theorem sum_first_six_primes_mod_seventh_prime :
  (first_six_primes.sum % seventh_prime) = 7 := by sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l632_63297


namespace NUMINAMATH_CALUDE_special_functions_bound_l632_63225

open Real

/-- Two differentiable real functions satisfying the given conditions -/
structure SpecialFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  hf : Differentiable ℝ f
  hg : Differentiable ℝ g
  h_eq : ∀ x, deriv f x / deriv g x = exp (f x - g x)
  h_f0 : f 0 = 1
  h_g2003 : g 2003 = 1

/-- The theorem stating that f(2003) > 1 - ln(2) for any pair of functions satisfying the conditions,
    and that 1 - ln(2) is the largest such constant -/
theorem special_functions_bound (sf : SpecialFunctions) :
  sf.f 2003 > 1 - log 2 ∧ ∀ c, (∀ sf' : SpecialFunctions, sf'.f 2003 > c) → c ≤ 1 - log 2 := by
  sorry

end NUMINAMATH_CALUDE_special_functions_bound_l632_63225


namespace NUMINAMATH_CALUDE_largest_square_tile_l632_63206

theorem largest_square_tile (a b : ℕ) (ha : a = 72) (hb : b = 90) :
  ∃ (s : ℕ), s = Nat.gcd a b ∧ 
  s * (a / s) = a ∧ 
  s * (b / s) = b ∧
  ∀ (t : ℕ), t * (a / t) = a → t * (b / t) = b → t ≤ s :=
sorry

end NUMINAMATH_CALUDE_largest_square_tile_l632_63206


namespace NUMINAMATH_CALUDE_root_ratio_sum_l632_63237

theorem root_ratio_sum (k₁ k₂ : ℝ) : 
  (∃ a b : ℝ, 3 * a^2 - (3 - k₁) * a + 7 = 0 ∧ 
              3 * b^2 - (3 - k₁) * b + 7 = 0 ∧ 
              a / b + b / a = 9 / 7) ∧
  (∃ a b : ℝ, 3 * a^2 - (3 - k₂) * a + 7 = 0 ∧ 
              3 * b^2 - (3 - k₂) * b + 7 = 0 ∧ 
              a / b + b / a = 9 / 7) →
  k₁ / k₂ + k₂ / k₁ = -20 / 7 := by
sorry

end NUMINAMATH_CALUDE_root_ratio_sum_l632_63237


namespace NUMINAMATH_CALUDE_inequality_condition_l632_63219

theorem inequality_condition (a b : ℝ) : 
  (∀ x : ℝ, (a + 1) * x^2 + a * x + a > b * (x^2 + x + 1)) ↔ b < a :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l632_63219


namespace NUMINAMATH_CALUDE_polynomial_equality_l632_63259

/-- Given a polynomial function q(x) satisfying the equation
    q(x) + (x^6 + 2x^4 + 5x^2 + 8x) = (3x^4 + 18x^3 + 20x^2 + 5x + 2),
    prove that q(x) = -x^6 + x^4 + 18x^3 + 15x^2 - 3x + 2 -/
theorem polynomial_equality (q : ℝ → ℝ) :
  (∀ x, q x + (x^6 + 2*x^4 + 5*x^2 + 8*x) = (3*x^4 + 18*x^3 + 20*x^2 + 5*x + 2)) →
  (∀ x, q x = -x^6 + x^4 + 18*x^3 + 15*x^2 - 3*x + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l632_63259


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l632_63242

def f (a b x : ℝ) : ℝ := x^5 - 3*x^4 + a*x^3 + b*x^2 - 5*x - 5

theorem polynomial_divisibility (a b : ℝ) :
  (∀ x, (x^2 - 1) ∣ f a b x) ↔ (a = 4 ∧ b = 8) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l632_63242


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l632_63257

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 280

/-- A debt that can be resolved using pigs and goats -/
def resolvable_debt (d : ℕ) : Prop :=
  ∃ (p g : ℤ), d = pig_value * p + goat_value * g

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 40

theorem smallest_resolvable_debt_is_correct :
  (resolvable_debt smallest_resolvable_debt) ∧
  (∀ d : ℕ, d > 0 ∧ d < smallest_resolvable_debt → ¬(resolvable_debt d)) :=
sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l632_63257


namespace NUMINAMATH_CALUDE_max_distance_complex_l632_63269

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  (⨆ z, |(2 + 3*I)*z^2 - z^4|) = 81 + 9 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_max_distance_complex_l632_63269


namespace NUMINAMATH_CALUDE_min_value_inequality_min_value_achievable_l632_63283

theorem min_value_inequality (a b c d : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 + (5/d - 1)^2 ≥ 5^(5/4) - 10*5^(1/4) + 5 :=
by sorry

theorem min_value_achievable : 
  ∃ (a b c d : ℝ), 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ 5 ∧
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 + (5/d - 1)^2 = 5^(5/4) - 10*5^(1/4) + 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_min_value_achievable_l632_63283


namespace NUMINAMATH_CALUDE_two_hour_walk_distance_l632_63233

/-- Calculates the total distance walked in two hours given the distance walked in the first hour -/
def total_distance (first_hour_distance : ℝ) : ℝ :=
  first_hour_distance + 2 * first_hour_distance

/-- Theorem stating that walking 2 km in the first hour and twice that in the second hour results in 6 km total -/
theorem two_hour_walk_distance :
  total_distance 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_hour_walk_distance_l632_63233


namespace NUMINAMATH_CALUDE_circle_intersection_chord_l632_63240

/-- Given two circles C₁ and C₂, where C₁ passes through the center of C₂,
    the equation of their chord of intersection is 5x + y - 19 = 0 -/
theorem circle_intersection_chord 
  (C₁ : ℝ → ℝ → Prop) 
  (C₂ : ℝ → ℝ → Prop) 
  (h₁ : ∀ x y, C₁ x y ↔ (x + 1)^2 + y^2 = r^2) 
  (h₂ : ∀ x y, C₂ x y ↔ (x - 4)^2 + (y - 1)^2 = 4) 
  (h₃ : C₁ 4 1) :
  ∀ x y, (C₁ x y ∧ C₂ x y) ↔ 5*x + y - 19 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_chord_l632_63240


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l632_63266

noncomputable def f (x : ℝ) : ℝ := -2 * x + x^3

theorem monotonic_decreasing_interval_of_f :
  ∀ x : ℝ, (x > -Real.sqrt 6 / 3 ∧ x < Real.sqrt 6 / 3) ↔ 
    StrictMonoOn f (Set.Ioo (-Real.sqrt 6 / 3) (Real.sqrt 6 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l632_63266


namespace NUMINAMATH_CALUDE_inverse_proportion_min_value_l632_63227

/-- Given an inverse proportion function y = k/x, prove that if the maximum value of y is 4
    when -2 ≤ x ≤ -1, then the minimum value of y is -1/2 when x ≥ 8 -/
theorem inverse_proportion_min_value (k : ℝ) :
  (∀ x, -2 ≤ x → x ≤ -1 → k / x ≤ 4) →
  (∃ x, -2 ≤ x ∧ x ≤ -1 ∧ k / x = 4) →
  (∀ x, x ≥ 8 → k / x ≥ -1/2) ∧
  (∃ x, x ≥ 8 ∧ k / x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_min_value_l632_63227


namespace NUMINAMATH_CALUDE_f_value_at_5_l632_63249

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 - b * x + 2

theorem f_value_at_5 (a b : ℝ) :
  f a b (-5) = 17 → f a b 5 = -13 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_5_l632_63249


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l632_63273

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The theorem stating that if S_2 = 3 and S_3 = 3, then S_5 = 0 for an arithmetic sequence -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h2 : seq.S 2 = 3) (h3 : seq.S 3 = 3) : seq.S 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l632_63273


namespace NUMINAMATH_CALUDE_kim_nail_polishes_l632_63264

/-- Given information about nail polishes owned by Kim, Heidi, and Karen, prove that Kim has 12 nail polishes. -/
theorem kim_nail_polishes :
  ∀ (K : ℕ), -- Kim's nail polishes
  (K + 5) + (K - 4) = 25 → -- Heidi and Karen's total
  K = 12 := by
sorry

end NUMINAMATH_CALUDE_kim_nail_polishes_l632_63264


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l632_63280

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x - a > -3) → a ∈ Set.Ioo (-6) 2 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l632_63280


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l632_63205

/-- Proves that adding 70 ounces of 60% salt solution to 70 ounces of 20% salt solution results in a 40% salt solution -/
theorem salt_solution_mixture : 
  let initial_volume : ℝ := 70
  let initial_concentration : ℝ := 0.2
  let added_volume : ℝ := 70
  let added_concentration : ℝ := 0.6
  let final_concentration : ℝ := 0.4
  (initial_volume * initial_concentration + added_volume * added_concentration) / (initial_volume + added_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l632_63205


namespace NUMINAMATH_CALUDE_matrix_not_invertible_iff_l632_63244

def matrix (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2*x, 5],
    ![4*x, 9]]

theorem matrix_not_invertible_iff (x : ℝ) :
  ¬(Matrix.det (matrix x) ≠ 0) ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_iff_l632_63244


namespace NUMINAMATH_CALUDE_franks_total_work_hours_l632_63287

/-- Calculates the total hours worked given the number of hours per day and number of days --/
def totalHours (hoursPerDay : ℕ) (numDays : ℕ) : ℕ :=
  hoursPerDay * numDays

/-- Theorem: Frank's total work hours --/
theorem franks_total_work_hours :
  totalHours 8 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_franks_total_work_hours_l632_63287


namespace NUMINAMATH_CALUDE_find_x_value_l632_63226

theorem find_x_value (A B : Set ℝ) (x : ℝ) : 
  A = {-1, 0} →
  B = {0, 1, x + 2} →
  A ⊆ B →
  x = -3 := by
sorry

end NUMINAMATH_CALUDE_find_x_value_l632_63226


namespace NUMINAMATH_CALUDE_mean_median_difference_l632_63250

/-- Represents the frequency distribution of days missed by students -/
def frequency_distribution : List (Nat × Nat) := [
  (0, 4),  -- 4 students missed 0 days
  (1, 2),  -- 2 students missed 1 day
  (2, 5),  -- 5 students missed 2 days
  (3, 2),  -- 2 students missed 3 days
  (4, 1),  -- 1 student missed 4 days
  (5, 3),  -- 3 students missed 5 days
  (6, 1)   -- 1 student missed 6 days
]

/-- Calculate the median of the distribution -/
def median (dist : List (Nat × Nat)) : Nat :=
  sorry

/-- Calculate the mean of the distribution -/
def mean (dist : List (Nat × Nat)) : Rat :=
  sorry

/-- The total number of students -/
def total_students : Nat := frequency_distribution.map (·.2) |>.sum

theorem mean_median_difference :
  mean frequency_distribution - median frequency_distribution = 0 ∧ total_students = 18 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l632_63250


namespace NUMINAMATH_CALUDE_triangle_8_6_4_l632_63221

/-- A triangle can be formed if the sum of any two sides is greater than the third side,
    and the difference between any two sides is less than the third side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  a - b < c ∧ b - c < a ∧ c - a < b

/-- Prove that line segments of lengths 8, 6, and 4 can form a triangle. -/
theorem triangle_8_6_4 : can_form_triangle 8 6 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_8_6_4_l632_63221


namespace NUMINAMATH_CALUDE_reflection_squared_is_identity_l632_63223

/-- A reflection matrix over a non-zero vector -/
def reflection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  sorry

/-- The identity matrix -/
def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  sorry

/-- Theorem: The square of a reflection matrix is the identity matrix -/
theorem reflection_squared_is_identity (v : ℝ × ℝ) (h : v ≠ (0, 0)) :
  (reflection_matrix v) ^ 2 = identity_matrix :=
sorry

end NUMINAMATH_CALUDE_reflection_squared_is_identity_l632_63223


namespace NUMINAMATH_CALUDE_visible_black_area_ratio_l632_63245

/-- Represents the area of a circle -/
structure CircleArea where
  area : ℝ
  area_pos : area > 0

/-- Represents the configuration of three circles -/
structure CircleConfiguration where
  black : CircleArea
  grey : CircleArea
  white : CircleArea
  initial_visible_black : ℝ
  final_visible_black : ℝ
  initial_condition : initial_visible_black = 7 * white.area
  final_condition : final_visible_black = initial_visible_black - white.area

/-- The theorem stating the ratio of visible black areas before and after rearrangement -/
theorem visible_black_area_ratio (config : CircleConfiguration) :
  config.initial_visible_black / config.final_visible_black = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_visible_black_area_ratio_l632_63245


namespace NUMINAMATH_CALUDE_proper_subsets_count_l632_63214

def S : Finset ℕ := {0, 3, 4}

theorem proper_subsets_count : (Finset.powerset S).card - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_proper_subsets_count_l632_63214


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_given_difference_and_larger_l632_63270

theorem sum_of_numbers_with_given_difference_and_larger (L S : ℤ) : 
  L = 35 → L - S = 15 → L + S = 55 := by sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_given_difference_and_larger_l632_63270


namespace NUMINAMATH_CALUDE_triangle_area_l632_63279

/-- Given a point A(a, 0) where a > 0, a line with 30° inclination tangent to circle O: x^2 + y^2 = r^2 
    at point B, and |AB| = √3, prove that the area of triangle OAB is √3/2 -/
theorem triangle_area (a r : ℝ) (ha : a > 0) (hr : r > 0) : 
  let A : ℝ × ℝ := (a, 0)
  let O : ℝ × ℝ := (0, 0)
  let line_slope : ℝ := Real.sqrt 3 / 3
  let circle (x y : ℝ) := x^2 + y^2 = r^2
  let tangent_line (x y : ℝ) := y = line_slope * (x - a)
  ∃ (B : ℝ × ℝ), 
    circle B.1 B.2 ∧ 
    tangent_line B.1 B.2 ∧ 
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = 3 →
    (1/2 : ℝ) * r * Real.sqrt 3 = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l632_63279


namespace NUMINAMATH_CALUDE_journey_ratio_theorem_l632_63298

/-- Represents the distance between two towns -/
structure Distance where
  miles : ℝ
  nonneg : miles ≥ 0

/-- Represents the speed of travel -/
structure Speed where
  mph : ℝ
  positive : mph > 0

/-- Represents a journey between two towns -/
structure Journey where
  distance : Distance
  speed : Speed

theorem journey_ratio_theorem 
  (speed_AB : Speed) 
  (speed_BC : Speed) 
  (avg_speed : Speed) 
  (h1 : speed_AB.mph = 60)
  (h2 : speed_BC.mph = 20)
  (h3 : avg_speed.mph = 36) :
  ∃ (dist_AB dist_BC : Distance),
    let journey_AB : Journey := ⟨dist_AB, speed_AB⟩
    let journey_BC : Journey := ⟨dist_BC, speed_BC⟩
    let total_distance : Distance := ⟨dist_AB.miles + dist_BC.miles, by sorry⟩
    let total_time : ℝ := dist_AB.miles / speed_AB.mph + dist_BC.miles / speed_BC.mph
    avg_speed.mph = total_distance.miles / total_time →
    dist_AB.miles / dist_BC.miles = 2 :=
by sorry

end NUMINAMATH_CALUDE_journey_ratio_theorem_l632_63298


namespace NUMINAMATH_CALUDE_cross_in_square_l632_63286

theorem cross_in_square (s : ℝ) : 
  s > 0 → 
  (2 * (s/2)^2 + 2 * (s/4)^2 = 810) → 
  s = 36 := by
sorry

end NUMINAMATH_CALUDE_cross_in_square_l632_63286


namespace NUMINAMATH_CALUDE_complement_of_complement_is_A_l632_63215

-- Define the universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- Define the complement of A in U
def C_UA : Set ℕ := {5, 7}

-- Define set A
def A : Set ℕ := {1, 3, 9}

-- Theorem statement
theorem complement_of_complement_is_A :
  A = U \ C_UA :=
by sorry

end NUMINAMATH_CALUDE_complement_of_complement_is_A_l632_63215


namespace NUMINAMATH_CALUDE_is_center_of_hyperbola_l632_63218

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 900 = 0

/-- The center of a hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 5)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem is_center_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_equation x y ↔ 
    ((y - hyperbola_center.2)^2 / (819/36) - (x - hyperbola_center.1)^2 / (819/9) = 1) :=
sorry

end NUMINAMATH_CALUDE_is_center_of_hyperbola_l632_63218


namespace NUMINAMATH_CALUDE_concatenated_number_divisible_by_1980_l632_63295

def concatenated_number : ℕ :=
  -- Definition of the number A as described in the problem
  sorry

theorem concatenated_number_divisible_by_1980 :
  1980 ∣ concatenated_number :=
by
  sorry

end NUMINAMATH_CALUDE_concatenated_number_divisible_by_1980_l632_63295


namespace NUMINAMATH_CALUDE_smallest_integer_larger_than_root_sum_eighth_power_l632_63247

theorem smallest_integer_larger_than_root_sum_eighth_power :
  ∃ n : ℤ, n = 1631 ∧ (∀ m : ℤ, m > (Real.sqrt 5 + Real.sqrt 3)^8 → m ≥ n) ∧
  (n - 1 : ℝ) ≤ (Real.sqrt 5 + Real.sqrt 3)^8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_larger_than_root_sum_eighth_power_l632_63247


namespace NUMINAMATH_CALUDE_ln_inequality_condition_l632_63228

theorem ln_inequality_condition (x : ℝ) :
  (∀ x, (Real.log x < 0 → x < 1)) ∧
  (∃ x, x < 1 ∧ Real.log x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ln_inequality_condition_l632_63228


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l632_63246

theorem difference_of_squares_example : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l632_63246


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l632_63200

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x + a| - x - 2

-- Theorem for part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 0} = {x : ℝ | x < 0 ∨ x > 2} := by sorry

-- Theorem for part II
theorem range_of_a (a : ℝ) (h : a > -1) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-a) 1 ∧ f a x₀ ≤ 0) →
  a ∈ Set.Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l632_63200


namespace NUMINAMATH_CALUDE_difference_divisible_by_nine_l632_63210

theorem difference_divisible_by_nine (a b : ℤ) : 
  ∃ k : ℤ, (3 * a + 2)^2 - (3 * b + 2)^2 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_difference_divisible_by_nine_l632_63210


namespace NUMINAMATH_CALUDE_corral_area_ratio_l632_63254

/-- The ratio of areas between four small square corrals and one large square corral -/
theorem corral_area_ratio (s : ℝ) (h : s > 0) : 
  (4 * s^2) / ((4 * s)^2) = 1 / 4 := by
  sorry

#check corral_area_ratio

end NUMINAMATH_CALUDE_corral_area_ratio_l632_63254


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sum_l632_63235

/-- A geometric series with the given property -/
def geometric_series (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- An arithmetic series -/
def arithmetic_series (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_arithmetic_sum (a b : ℕ → ℝ) :
  geometric_series a →
  arithmetic_series b →
  a 3 * a 11 = 4 * a 7 →
  b 7 = a 7 →
  b 5 + b 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sum_l632_63235


namespace NUMINAMATH_CALUDE_horner_v3_eq_7_9_l632_63213

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 2x^4 + 3.5x^3 - 2.6x^2 + 1.7x - 0.8 -/
def f : ℝ → ℝ := fun x => 5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- Coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [-0.8, 1.7, -2.6, 3.5, 2, 5]

/-- Theorem: Horner's method for f(x) at x = 1 gives v₃ = 7.9 -/
theorem horner_v3_eq_7_9 : 
  (horner (coeffs.take 4) 1) = 7.9 := by sorry

end NUMINAMATH_CALUDE_horner_v3_eq_7_9_l632_63213


namespace NUMINAMATH_CALUDE_smallest_n_inequality_l632_63231

theorem smallest_n_inequality (w x y z : ℝ) : 
  ∃ (n : ℕ), (w^2 + x^2 + y^2 + z^2)^2 ≤ n*(w^4 + x^4 + y^4 + z^4) ∧ 
  ∀ (m : ℕ), m < n → ∃ (a b c d : ℝ), (a^2 + b^2 + c^2 + d^2)^2 > m*(a^4 + b^4 + c^4 + d^4) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_inequality_l632_63231


namespace NUMINAMATH_CALUDE_factorization_equality_l632_63290

theorem factorization_equality (x : ℝ) : 75 * x^3 - 225 * x^10 = 75 * x^3 * (1 - 3 * x^7) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l632_63290


namespace NUMINAMATH_CALUDE_exists_consecutive_numbers_with_54_times_product_l632_63220

def nonZeroDigits (n : ℕ) : List ℕ :=
  (n.digits 10).filter (· ≠ 0)

def productOfNonZeroDigits (n : ℕ) : ℕ :=
  (nonZeroDigits n).prod

theorem exists_consecutive_numbers_with_54_times_product : 
  ∃ n : ℕ, productOfNonZeroDigits (n + 1) = 54 * productOfNonZeroDigits n := by
  sorry

end NUMINAMATH_CALUDE_exists_consecutive_numbers_with_54_times_product_l632_63220


namespace NUMINAMATH_CALUDE_parallel_vectors_x_equals_one_l632_63258

/-- Given two parallel vectors a and b, prove that x = 1 -/
theorem parallel_vectors_x_equals_one (x : ℝ) :
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (2, 4)
  (∃ (k : ℝ), a = k • b) →
  x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_equals_one_l632_63258


namespace NUMINAMATH_CALUDE_subset_count_divisible_by_prime_l632_63265

theorem subset_count_divisible_by_prime (p : Nat) (hp : Nat.Prime p) (hp_odd : Odd p) :
  let S := Finset.range (2 * p)
  (Finset.filter (fun A : Finset Nat =>
    A.card = p ∧ (A.sum id) % p = 0) (Finset.powerset S)).card =
  (1 / p) * (Nat.choose (2 * p) p - 2) + 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_count_divisible_by_prime_l632_63265


namespace NUMINAMATH_CALUDE_rotary_club_eggs_l632_63284

/-- Calculates the total number of eggs needed for the Rotary Club's Omelet Breakfast --/
def total_eggs_needed (small_children : ℕ) (older_children : ℕ) (adults : ℕ) (seniors : ℕ) 
  (waste_percent : ℚ) (extra_omelets : ℕ) (eggs_per_extra_omelet : ℚ) : ℕ :=
  let eggs_for_tickets := small_children + 2 * older_children + 3 * adults + 4 * seniors
  let waste_eggs := ⌈(eggs_for_tickets : ℚ) * waste_percent⌉
  let extra_omelet_eggs := ⌈(extra_omelets : ℚ) * eggs_per_extra_omelet⌉
  eggs_for_tickets + waste_eggs.toNat + extra_omelet_eggs.toNat

/-- Theorem stating the total number of eggs needed for the Rotary Club's Omelet Breakfast --/
theorem rotary_club_eggs : 
  total_eggs_needed 53 35 75 37 (3/100) 25 (5/2) = 574 := by
  sorry

end NUMINAMATH_CALUDE_rotary_club_eggs_l632_63284


namespace NUMINAMATH_CALUDE_triangle_side_length_l632_63211

/-- 
Given a triangle ABC where:
- a, b, c are sides opposite to angles A, B, C respectively
- A = 2π/3
- b = √2
- Area of triangle ABC is √3
Prove that a = √14
-/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 2 * Real.pi / 3 →
  b = Real.sqrt 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a = Real.sqrt 14 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l632_63211


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l632_63216

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the r-th term in the expansion of (1+2x)^7 -/
def coefficient (r : ℕ) : ℕ := binomial 7 r * 2^r

theorem binomial_expansion_properties :
  (coefficient 2 = binomial 7 2 * 2^2) ∧
  (coefficient 2 = 24) := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l632_63216


namespace NUMINAMATH_CALUDE_distance_to_directrix_l632_63256

/-- The distance from a point on a parabola to its directrix -/
theorem distance_to_directrix (p : ℝ) (h : p > 0) : 
  let A : ℝ × ℝ := (1, Real.sqrt 5)
  let C := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  A ∈ C → |1 - (-p/2)| = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_directrix_l632_63256


namespace NUMINAMATH_CALUDE_modified_baseball_league_games_l632_63260

/-- The total number of games played in a modified baseball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team,
    the total number of games played is 180 -/
theorem modified_baseball_league_games :
  total_games 10 4 = 180 := by
  sorry

#eval total_games 10 4

end NUMINAMATH_CALUDE_modified_baseball_league_games_l632_63260
