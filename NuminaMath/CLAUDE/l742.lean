import Mathlib

namespace NUMINAMATH_CALUDE_james_new_weight_l742_74279

/-- Calculates the new weight after muscle and fat gain -/
def new_weight (initial_weight : ℝ) (muscle_gain_percentage : ℝ) (fat_gain_ratio : ℝ) : ℝ :=
  let muscle_gain := initial_weight * muscle_gain_percentage
  let fat_gain := muscle_gain * fat_gain_ratio
  initial_weight + muscle_gain + fat_gain

/-- Proves that James's new weight is 150 kg after gaining muscle and fat -/
theorem james_new_weight :
  new_weight 120 0.2 0.25 = 150 := by
  sorry

end NUMINAMATH_CALUDE_james_new_weight_l742_74279


namespace NUMINAMATH_CALUDE_percentage_enrolled_in_biology_l742_74237

theorem percentage_enrolled_in_biology (total_students : ℕ) (not_enrolled : ℕ) 
  (h1 : total_students = 880) (h2 : not_enrolled = 638) :
  (((total_students - not_enrolled) : ℚ) / total_students) * 100 = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_enrolled_in_biology_l742_74237


namespace NUMINAMATH_CALUDE_sine_arithmetic_sequence_l742_74255

open Real

theorem sine_arithmetic_sequence (a : ℝ) :
  0 < a ∧ a < 2 * π →
  (sin a + sin (3 * a) = 2 * sin (2 * a)) ↔ (a = π / 2 ∨ a = 3 * π / 2) :=
by sorry

end NUMINAMATH_CALUDE_sine_arithmetic_sequence_l742_74255


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l742_74247

theorem final_sum_after_transformation (S x k : ℝ) (a b : ℝ) (h : a + b = S) :
  k * (a + x) + k * (b + x) = k * S + 2 * k * x := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l742_74247


namespace NUMINAMATH_CALUDE_inequality_system_solution_l742_74203

/-- Proves that the solution set of the given inequality system is (-2, 1]. -/
theorem inequality_system_solution :
  ∀ x : ℝ, (x > -6 - 2*x ∧ x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l742_74203


namespace NUMINAMATH_CALUDE_cubic_three_zeros_l742_74249

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem cubic_three_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔ 
  a < -3 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_zeros_l742_74249


namespace NUMINAMATH_CALUDE_train_length_l742_74213

/-- Given a train crossing a bridge, calculate its length -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 225 →
  crossing_time = 30 →
  train_speed * crossing_time - bridge_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l742_74213


namespace NUMINAMATH_CALUDE_age_ratio_proof_l742_74265

def arun_future_age : ℕ := 26
def years_to_future : ℕ := 6
def deepak_current_age : ℕ := 15

theorem age_ratio_proof :
  let arun_current_age := arun_future_age - years_to_future
  (arun_current_age : ℚ) / deepak_current_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l742_74265


namespace NUMINAMATH_CALUDE_convex_pentagon_angles_obtuse_l742_74206

/-- A convex pentagon with equal sides and each angle less than 120° -/
structure ConvexPentagon where
  -- The pentagon is convex
  is_convex : Bool
  -- All sides are equal
  equal_sides : Bool
  -- Each angle is less than 120°
  angles_less_than_120 : Bool

/-- Theorem: In a convex pentagon with equal sides and each angle less than 120°, 
    each angle is greater than 90° -/
theorem convex_pentagon_angles_obtuse (p : ConvexPentagon) : 
  p.is_convex ∧ p.equal_sides ∧ p.angles_less_than_120 → 
  ∀ angle, angle > 90 := by sorry

end NUMINAMATH_CALUDE_convex_pentagon_angles_obtuse_l742_74206


namespace NUMINAMATH_CALUDE_g_of_one_eq_neg_two_l742_74217

theorem g_of_one_eq_neg_two :
  let g : ℝ → ℝ := fun x ↦ x^3 - x^2 - 2*x
  g 1 = -2 := by sorry

end NUMINAMATH_CALUDE_g_of_one_eq_neg_two_l742_74217


namespace NUMINAMATH_CALUDE_hill_climbing_speed_l742_74253

/-- Proves that given a round trip journey with specified conditions, 
    the average speed for the upward journey is 2.625 km/h -/
theorem hill_climbing_speed 
  (upward_time : ℝ) 
  (downward_time : ℝ) 
  (total_avg_speed : ℝ) 
  (h1 : upward_time = 4) 
  (h2 : downward_time = 2) 
  (h3 : total_avg_speed = 3.5) : 
  (total_avg_speed * (upward_time + downward_time)) / (2 * upward_time) = 2.625 := by
sorry

end NUMINAMATH_CALUDE_hill_climbing_speed_l742_74253


namespace NUMINAMATH_CALUDE_price_increase_l742_74294

theorem price_increase (P : ℝ) (x : ℝ) (h1 : P > 0) :
  1.25 * P * (1 + x / 100) = 1.625 * P → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_l742_74294


namespace NUMINAMATH_CALUDE_fraction_of_students_with_partners_l742_74223

theorem fraction_of_students_with_partners :
  ∀ (a b : ℕ), 
    a > 0 → b > 0 →
    (b : ℚ) / 4 = (3 : ℚ) * a / 7 →
    ((b : ℚ) / 4 + (3 : ℚ) * a / 7) / ((b : ℚ) + a) = 6 / 19 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_students_with_partners_l742_74223


namespace NUMINAMATH_CALUDE_triangle_angle_sine_relation_l742_74242

theorem triangle_angle_sine_relation (A B : Real) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) :
  A > B ↔ Real.sin A > Real.sin B := by sorry

end NUMINAMATH_CALUDE_triangle_angle_sine_relation_l742_74242


namespace NUMINAMATH_CALUDE_matchstick_length_theorem_l742_74287

/-- Represents a figure made of matchsticks -/
structure MatchstickFigure where
  smallSquareCount : ℕ
  largeSquareCount : ℕ
  totalArea : ℝ

/-- Calculates the total length of matchsticks used in the figure -/
def totalMatchstickLength (figure : MatchstickFigure) : ℝ :=
  sorry

/-- Theorem stating the total length of matchsticks in the given figure -/
theorem matchstick_length_theorem (figure : MatchstickFigure) 
  (h1 : figure.smallSquareCount = 8)
  (h2 : figure.largeSquareCount = 1)
  (h3 : figure.totalArea = 300) :
  totalMatchstickLength figure = 140 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_length_theorem_l742_74287


namespace NUMINAMATH_CALUDE_mod7_mul_table_mod10_mul_2_mod10_mul_5_mod9_mul_3_l742_74282

-- Define the modular multiplication function
def modMul (a b m : Nat) : Nat :=
  (a * b) % m

-- Theorem for modulo 7 multiplication table
theorem mod7_mul_table (a b : Fin 7) : 
  modMul a b 7 = 
    match a, b with
    | 0, _ => 0
    | _, 0 => 0
    | 1, x => x
    | x, 1 => x
    | 2, 2 => 4
    | 2, 3 => 6
    | 2, 4 => 1
    | 2, 5 => 3
    | 2, 6 => 5
    | 3, 2 => 6
    | 3, 3 => 2
    | 3, 4 => 5
    | 3, 5 => 1
    | 3, 6 => 4
    | 4, 2 => 1
    | 4, 3 => 5
    | 4, 4 => 2
    | 4, 5 => 6
    | 4, 6 => 3
    | 5, 2 => 3
    | 5, 3 => 1
    | 5, 4 => 6
    | 5, 5 => 4
    | 5, 6 => 2
    | 6, 2 => 5
    | 6, 3 => 4
    | 6, 4 => 3
    | 6, 5 => 2
    | 6, 6 => 1
    | _, _ => 0  -- This case should never be reached
  := by sorry

-- Theorem for modulo 10 multiplication by 2
theorem mod10_mul_2 (a : Fin 10) : 
  modMul 2 a 10 = 
    match a with
    | 0 => 0
    | 1 => 2
    | 2 => 4
    | 3 => 6
    | 4 => 8
    | 5 => 0
    | 6 => 2
    | 7 => 4
    | 8 => 6
    | 9 => 8
  := by sorry

-- Theorem for modulo 10 multiplication by 5
theorem mod10_mul_5 (a : Fin 10) : 
  modMul 5 a 10 = 
    match a with
    | 0 => 0
    | 1 => 5
    | 2 => 0
    | 3 => 5
    | 4 => 0
    | 5 => 5
    | 6 => 0
    | 7 => 5
    | 8 => 0
    | 9 => 5
  := by sorry

-- Theorem for modulo 9 multiplication by 3
theorem mod9_mul_3 (a : Fin 9) : 
  modMul 3 a 9 = 
    match a with
    | 0 => 0
    | 1 => 3
    | 2 => 6
    | 3 => 0
    | 4 => 3
    | 5 => 6
    | 6 => 0
    | 7 => 3
    | 8 => 6
  := by sorry

end NUMINAMATH_CALUDE_mod7_mul_table_mod10_mul_2_mod10_mul_5_mod9_mul_3_l742_74282


namespace NUMINAMATH_CALUDE_age_problem_l742_74228

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 22 →
  b = 8 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l742_74228


namespace NUMINAMATH_CALUDE_floor_sqrt_50_l742_74221

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_l742_74221


namespace NUMINAMATH_CALUDE_basketball_time_calculation_l742_74283

def football_time : ℕ := 60
def total_time_hours : ℕ := 2

theorem basketball_time_calculation :
  football_time + (total_time_hours * 60 - football_time) = 60 := by
  sorry

end NUMINAMATH_CALUDE_basketball_time_calculation_l742_74283


namespace NUMINAMATH_CALUDE_sum_nonpositive_implies_one_nonpositive_l742_74225

theorem sum_nonpositive_implies_one_nonpositive (x y : ℝ) : 
  x + y ≤ 0 → x ≤ 0 ∨ y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_nonpositive_implies_one_nonpositive_l742_74225


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l742_74256

def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | -2 < x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = {x | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l742_74256


namespace NUMINAMATH_CALUDE_specific_pyramid_volume_l742_74260

/-- A right pyramid with a square base -/
structure SquarePyramid where
  base_area : ℝ
  total_surface_area : ℝ
  triangular_face_area : ℝ

/-- The volume of a square pyramid -/
def volume (p : SquarePyramid) : ℝ := sorry

/-- Theorem stating the volume of the specific pyramid -/
theorem specific_pyramid_volume :
  ∀ (p : SquarePyramid),
  p.total_surface_area = 648 ∧
  p.triangular_face_area = (1/3) * p.base_area ∧
  p.total_surface_area = p.base_area + 4 * p.triangular_face_area →
  volume p = (4232 * Real.sqrt 6) / 9 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_volume_l742_74260


namespace NUMINAMATH_CALUDE_intersection_condition_distance_product_condition_l742_74296

-- Define the curve C in Cartesian coordinates
def C (x y : ℝ) : Prop := x^2 + y^2 = 2*x

-- Define the line l
def l (m t : ℝ) : ℝ × ℝ := (m + 3*t, 4*t)

-- Define the intersection condition
def intersects_at_two_points (m : ℝ) : Prop :=
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ C (l m t₁).1 (l m t₁).2 ∧ C (l m t₂).1 (l m t₂).2

-- Define the distance product condition
def distance_product_is_one (m : ℝ) : Prop :=
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ C (l m t₁).1 (l m t₁).2 ∧ C (l m t₂).1 (l m t₂).2 ∧
    (m^2 + (3*t₁)^2 + (4*t₁)^2) * (m^2 + (3*t₂)^2 + (4*t₂)^2) = 1

-- State the theorems
theorem intersection_condition (m : ℝ) :
  intersects_at_two_points m ↔ -1/4 < m ∧ m < 9/4 :=
sorry

theorem distance_product_condition :
  ∃ m, distance_product_is_one m ∧ m = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_distance_product_condition_l742_74296


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l742_74269

def scores : List ℝ := [93, 87, 90, 94, 88, 92]

theorem arithmetic_mean_of_scores :
  (scores.sum / scores.length : ℝ) = 90.6667 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l742_74269


namespace NUMINAMATH_CALUDE_fourth_quadrant_condition_negative_x_axis_condition_upper_half_plane_condition_l742_74248

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 + 3*m - 28)

-- Theorem for the fourth quadrant condition
theorem fourth_quadrant_condition (m : ℝ) :
  (z m).re > 0 ∧ (z m).im < 0 ↔ -7 < m ∧ m < 3 := by sorry

-- Theorem for the negative half of x-axis condition
theorem negative_x_axis_condition (m : ℝ) :
  (z m).re < 0 ∧ (z m).im = 0 ↔ m = 4 := by sorry

-- Theorem for the upper half-plane condition
theorem upper_half_plane_condition (m : ℝ) :
  (z m).im ≥ 0 ↔ m ≥ 4 ∨ m ≤ -7 := by sorry

end NUMINAMATH_CALUDE_fourth_quadrant_condition_negative_x_axis_condition_upper_half_plane_condition_l742_74248


namespace NUMINAMATH_CALUDE_unique_solution_condition_l742_74218

theorem unique_solution_condition (a : ℝ) : 
  (∃! x, 0 ≤ x^2 + a*x + 6 ∧ x^2 + a*x + 6 ≤ 4) ↔ (a = 2*Real.sqrt 2 ∨ a = -2*Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l742_74218


namespace NUMINAMATH_CALUDE_half_vector_MN_l742_74222

/-- Given two vectors OM and ON in ℝ², prove that half of vector MN is (1/2, -4) -/
theorem half_vector_MN (OM ON : ℝ × ℝ) (h1 : OM = (-2, 3)) (h2 : ON = (-1, -5)) :
  (1 / 2 : ℝ) • (ON - OM) = (1/2, -4) := by
  sorry

end NUMINAMATH_CALUDE_half_vector_MN_l742_74222


namespace NUMINAMATH_CALUDE_ad_rate_per_square_foot_l742_74264

-- Define the problem parameters
def num_companies : ℕ := 3
def ads_per_company : ℕ := 10
def ad_length : ℕ := 12
def ad_width : ℕ := 5
def total_paid : ℕ := 108000

-- Define the theorem
theorem ad_rate_per_square_foot :
  let total_area : ℕ := num_companies * ads_per_company * ad_length * ad_width
  let rate_per_square_foot : ℚ := total_paid / total_area
  rate_per_square_foot = 60 := by
  sorry

end NUMINAMATH_CALUDE_ad_rate_per_square_foot_l742_74264


namespace NUMINAMATH_CALUDE_unique_three_digit_cube_sum_l742_74286

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem unique_three_digit_cube_sum : ∃! n : ℕ, 
  is_three_digit_number n ∧ n = (digit_sum n)^3 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_cube_sum_l742_74286


namespace NUMINAMATH_CALUDE_set_intersection_equality_l742_74250

def M : Set ℤ := {1, 2, 3}
def N : Set ℤ := {x : ℤ | 1 < x ∧ x < 4}

theorem set_intersection_equality : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l742_74250


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l742_74275

theorem binomial_expansion_coefficient (p : ℝ) : 
  (∃ k : ℕ, Nat.choose 5 k * p^k = 80 ∧ 2*k = 6) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l742_74275


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l742_74276

theorem square_rectangle_area_relation :
  ∀ x : ℝ,
  let square_side : ℝ := x - 3
  let rect_length : ℝ := x - 4
  let rect_width : ℝ := x + 5
  let square_area : ℝ := square_side ^ 2
  let rect_area : ℝ := rect_length * rect_width
  (rect_area = 3 * square_area) →
  (∃ y : ℝ, y ≠ x ∧ 
    let square_side' : ℝ := y - 3
    let rect_length' : ℝ := y - 4
    let rect_width' : ℝ := y + 5
    let square_area' : ℝ := square_side' ^ 2
    let rect_area' : ℝ := rect_length' * rect_width'
    (rect_area' = 3 * square_area')) →
  x + y = 7 :=
by sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l742_74276


namespace NUMINAMATH_CALUDE_binary_100_is_4_binary_101_is_5_binary_1100_is_12_l742_74258

-- Define binary to decimal conversion function
def binaryToDecimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

-- Define decimal numbers
def four : ℕ := 4
def five : ℕ := 5
def twelve : ℕ := 12

-- Define binary numbers
def binary_100 : List Bool := [true, false, false]
def binary_101 : List Bool := [true, false, true]
def binary_1100 : List Bool := [true, true, false, false]

-- Theorem statements
theorem binary_100_is_4 : binaryToDecimal binary_100 = four := by sorry

theorem binary_101_is_5 : binaryToDecimal binary_101 = five := by sorry

theorem binary_1100_is_12 : binaryToDecimal binary_1100 = twelve := by sorry

end NUMINAMATH_CALUDE_binary_100_is_4_binary_101_is_5_binary_1100_is_12_l742_74258


namespace NUMINAMATH_CALUDE_pine_cones_on_roof_l742_74240

/-- The number of pine trees in Alan's backyard -/
def num_trees : ℕ := 8

/-- The number of pine cones dropped by each tree -/
def cones_per_tree : ℕ := 200

/-- The weight of each pine cone in ounces -/
def cone_weight : ℕ := 4

/-- The total weight of pine cones on Alan's roof in ounces -/
def roof_weight : ℕ := 1920

/-- The percentage of pine cones that fall on Alan's roof -/
def roof_percentage : ℚ := 30 / 100

theorem pine_cones_on_roof :
  (roof_weight / cone_weight) / (num_trees * cones_per_tree) = roof_percentage := by
  sorry

end NUMINAMATH_CALUDE_pine_cones_on_roof_l742_74240


namespace NUMINAMATH_CALUDE_exists_decreasing_arithmetic_with_non_decreasing_sums_l742_74238

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sequence of partial sums of a given sequence -/
def partial_sums (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => partial_sums a n + a (n + 1)

/-- A sequence is decreasing -/
def is_decreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ a n

theorem exists_decreasing_arithmetic_with_non_decreasing_sums :
  ∃ a : ℕ → ℝ,
    arithmetic_sequence a ∧
    is_decreasing a ∧
    (∀ n : ℕ, a n = -2 * n + 7) ∧
    ¬(is_decreasing (partial_sums a)) := by
  sorry

end NUMINAMATH_CALUDE_exists_decreasing_arithmetic_with_non_decreasing_sums_l742_74238


namespace NUMINAMATH_CALUDE_a_range_l742_74288

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + x^2 - a*x

theorem a_range (a : ℝ) :
  (∀ x > 0, Monotone (f a)) →
  (∀ x ∈ Set.Ioc 0 1, f a x ≤ 1/2 * (3*x^2 + 1/x^2 - 6*x)) →
  2 ≤ a ∧ a ≤ 2 * sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_a_range_l742_74288


namespace NUMINAMATH_CALUDE_card_sets_l742_74208

def is_valid_card_set (a b c d : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 9 ∧
  (([a + b, a + c, a + d, b + c, b + d, c + d].filter (· = 9)).length = 2) ∧
  (([a + b, a + c, a + d, b + c, b + d, c + d].filter (· < 9)).length = 2) ∧
  (([a + b, a + c, a + d, b + c, b + d, c + d].filter (· > 9)).length = 2)

theorem card_sets :
  ∀ a b c d : ℕ,
    is_valid_card_set a b c d ↔
      (a = 1 ∧ b = 2 ∧ c = 7 ∧ d = 8) ∨
      (a = 1 ∧ b = 3 ∧ c = 6 ∧ d = 8) ∨
      (a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 8) ∨
      (a = 2 ∧ b = 3 ∧ c = 6 ∧ d = 7) ∨
      (a = 2 ∧ b = 4 ∧ c = 5 ∧ d = 7) ∨
      (a = 3 ∧ b = 4 ∧ c = 5 ∧ d = 6) :=
by sorry

end NUMINAMATH_CALUDE_card_sets_l742_74208


namespace NUMINAMATH_CALUDE_james_tree_problem_l742_74284

/-- Represents the number of trees James initially has -/
def initial_trees : ℕ := 2

/-- Represents the percentage of seeds planted -/
def planting_rate : ℚ := 60 / 100

/-- Represents the number of new trees planted -/
def new_trees : ℕ := 24

/-- Represents the number of plants per tree -/
def plants_per_tree : ℕ := 20

theorem james_tree_problem :
  plants_per_tree * initial_trees * planting_rate = new_trees :=
sorry

end NUMINAMATH_CALUDE_james_tree_problem_l742_74284


namespace NUMINAMATH_CALUDE_triangle_formation_l742_74262

/-- Checks if three lengths can form a triangle according to the triangle inequality theorem -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  can_form_triangle 3 4 5 ∧
  ¬can_form_triangle 1 1 2 ∧
  ¬can_form_triangle 1 4 6 ∧
  ¬can_form_triangle 2 3 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l742_74262


namespace NUMINAMATH_CALUDE_f_plus_g_at_one_equals_two_l742_74257

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_plus_g_at_one_equals_two
  (f g : ℝ → ℝ)
  (h_even : isEven f)
  (h_odd : isOdd g)
  (h_eq : ∀ x, f x - g x = x^3 + x^2 + 1) :
  f 1 + g 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_f_plus_g_at_one_equals_two_l742_74257


namespace NUMINAMATH_CALUDE_min_good_pairs_l742_74267

/-- A circular arrangement of natural numbers from 1 to 100 -/
def CircularArrangement := Fin 100 → ℕ

/-- Predicate to check if a number at index i satisfies the neighbor condition -/
def satisfies_neighbor_condition (arr : CircularArrangement) (i : Fin 100) : Prop :=
  (arr i > arr ((i + 1) % 100) ∧ arr i > arr ((i + 99) % 100)) ∨
  (arr i < arr ((i + 1) % 100) ∧ arr i < arr ((i + 99) % 100))

/-- Predicate to check if a pair at indices i and j form a "good pair" -/
def is_good_pair (arr : CircularArrangement) (i j : Fin 100) : Prop :=
  arr i > arr j ∧ satisfies_neighbor_condition arr i ∧ satisfies_neighbor_condition arr j

/-- The main theorem stating that any valid arrangement has at least 51 good pairs -/
theorem min_good_pairs (arr : CircularArrangement) 
  (h_valid : ∀ i, satisfies_neighbor_condition arr i)
  (h_distinct : ∀ i j, i ≠ j → arr i ≠ arr j)
  (h_range : ∀ i, arr i ∈ Finset.range 101 \ {0}) :
  ∃ (pairs : Finset (Fin 100 × Fin 100)), pairs.card ≥ 51 ∧ 
    ∀ (p : Fin 100 × Fin 100), p ∈ pairs → is_good_pair arr p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_min_good_pairs_l742_74267


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l742_74200

theorem average_of_three_numbers (N : ℕ) : 
  15 < N ∧ N < 23 ∧ Even N → 
  (∃ x, x = (8 + 12 + N) / 3 ∧ (x = 12 ∨ x = 14)) :=
sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l742_74200


namespace NUMINAMATH_CALUDE_second_fish_length_is_02_l742_74273

/-- The length of the first fish in feet -/
def first_fish_length : ℝ := 0.3

/-- The difference in length between the first and second fish in feet -/
def length_difference : ℝ := 0.1

/-- The length of the second fish in feet -/
def second_fish_length : ℝ := first_fish_length - length_difference

/-- Theorem stating that the second fish is 0.2 foot long -/
theorem second_fish_length_is_02 : second_fish_length = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_second_fish_length_is_02_l742_74273


namespace NUMINAMATH_CALUDE_equation_solution_l742_74251

theorem equation_solution (x : ℤ) : 9*x + 2 ≡ 7 [ZMOD 15] ↔ x ≡ 10 [ZMOD 15] := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l742_74251


namespace NUMINAMATH_CALUDE_domain_of_f_l742_74233

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 5*x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -3 ∧ x ≠ -2} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_l742_74233


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l742_74209

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l742_74209


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l742_74205

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (max : ℝ), (∀ a b : ℝ, a^2 + b^2 = 4 → |a| + |b| ≤ max) ∧ (|x| + |y| = max) :=
by sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l742_74205


namespace NUMINAMATH_CALUDE_johns_friends_count_l742_74215

def total_cost : ℕ := 12100
def cost_per_person : ℕ := 1100

theorem johns_friends_count : 
  (total_cost / cost_per_person) - 1 = 10 := by sorry

end NUMINAMATH_CALUDE_johns_friends_count_l742_74215


namespace NUMINAMATH_CALUDE_no_integer_roots_l742_74280

theorem no_integer_roots (a b : ℤ) : 
  ¬ ∃ (x : ℤ), (x^2 + 10*a*x + 5*b + 3 = 0) ∨ (x^2 + 10*a*x + 5*b - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_roots_l742_74280


namespace NUMINAMATH_CALUDE_product_testing_theorem_l742_74245

/-- The number of ways to choose k items from n items, where order matters and repetition is not allowed. -/
def A (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of products -/
def total_products : ℕ := 10

/-- The number of defective products -/
def defective_products : ℕ := 4

/-- The number of ways to find 4 defective products among 10 products, 
    where the first defective is found on the 2nd measurement and the last on the 8th -/
def ways_specific_case : ℕ := A 4 2 * A 5 2 * A 6 4

/-- The number of ways to find 4 defective products among 10 products in at most 6 measurements -/
def ways_at_most_6 : ℕ := A 4 4 + 4 * A 4 3 * A 6 1 + 4 * A 5 3 * A 6 2 + A 6 6

theorem product_testing_theorem :
  (ways_specific_case = A 4 2 * A 5 2 * A 6 4) ∧
  (ways_at_most_6 = A 4 4 + 4 * A 4 3 * A 6 1 + 4 * A 5 3 * A 6 2 + A 6 6) :=
sorry

end NUMINAMATH_CALUDE_product_testing_theorem_l742_74245


namespace NUMINAMATH_CALUDE_min_value_expression_l742_74244

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  let A := (Real.sqrt (a^6 + b^4 * c^6)) / b + 
           (Real.sqrt (b^6 + c^4 * a^6)) / c + 
           (Real.sqrt (c^6 + a^4 * b^6)) / a
  A ≥ 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l742_74244


namespace NUMINAMATH_CALUDE_sneakers_final_price_l742_74229

/-- Calculates the final price of sneakers after applying a coupon and membership discount -/
theorem sneakers_final_price
  (original_price : ℝ)
  (coupon_discount : ℝ)
  (membership_discount_rate : ℝ)
  (h1 : original_price = 120)
  (h2 : coupon_discount = 10)
  (h3 : membership_discount_rate = 0.1) :
  let price_after_coupon := original_price - coupon_discount
  let membership_discount := price_after_coupon * membership_discount_rate
  let final_price := price_after_coupon - membership_discount
  final_price = 99 := by
sorry

end NUMINAMATH_CALUDE_sneakers_final_price_l742_74229


namespace NUMINAMATH_CALUDE_binary_addition_proof_l742_74202

def binary_to_nat : List Bool → Nat
  | [] => 0
  | b::bs => (if b then 1 else 0) + 2 * binary_to_nat bs

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then
    []
  else
    (n % 2 = 1) :: nat_to_binary (n / 2)

theorem binary_addition_proof :
  let a := [false, true, false, true]  -- 1010₂
  let b := [false, true]               -- 10₂
  let sum := [false, false, true, true] -- 1100₂
  binary_to_nat a + binary_to_nat b = binary_to_nat sum := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_proof_l742_74202


namespace NUMINAMATH_CALUDE_triangle_side_length_l742_74210

/-- Given a triangle ABC with side lengths and altitude, prove BC = 4 -/
theorem triangle_side_length (A B C : ℝ × ℝ) (h : ℝ) : 
  let d := (fun (P Q : ℝ × ℝ) => Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))
  (d A B = 2 * Real.sqrt 3) →
  (d A C = 2) →
  (h = Real.sqrt 3) →
  (h * d B C = 2 * Real.sqrt 3 * 2) →
  d B C = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l742_74210


namespace NUMINAMATH_CALUDE_intersecting_line_equation_l742_74212

/-- A line that intersects a circle and a hyperbola with specific properties -/
structure IntersectingLine (a : ℝ) where
  m : ℝ
  b : ℝ
  intersects_circle : ∀ x y, y = m * x + b → x^2 + y^2 = a^2
  intersects_hyperbola : ∀ x y, y = m * x + b → x^2 - y^2 = a^2
  trisects : ∀ (x₁ x₂ x₃ x₄ : ℝ),
    (x₁^2 + (m * x₁ + b)^2 = a^2) →
    (x₂^2 + (m * x₂ + b)^2 = a^2) →
    (x₃^2 - (m * x₃ + b)^2 = a^2) →
    (x₄^2 - (m * x₄ + b)^2 = a^2) →
    (x₁ - x₂)^2 = (1/9) * (x₃ - x₄)^2

/-- The equation of the intersecting line is y = ±(2√5/5)x or y = ±(2√5/5)a -/
theorem intersecting_line_equation (a : ℝ) (l : IntersectingLine a) :
  (l.m = 2 * Real.sqrt 5 / 5 ∧ l.b = 0) ∨
  (l.m = -2 * Real.sqrt 5 / 5 ∧ l.b = 0) ∨
  (l.m = 0 ∧ l.b = 2 * a * Real.sqrt 5 / 5) ∨
  (l.m = 0 ∧ l.b = -2 * a * Real.sqrt 5 / 5) :=
sorry

end NUMINAMATH_CALUDE_intersecting_line_equation_l742_74212


namespace NUMINAMATH_CALUDE_fifteen_balls_four_draws_l742_74292

/-- The number of ways to draw n balls in order from a bin of m balls,
    where each ball remains outside the bin after it is drawn. -/
def orderedDraw (m n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * (m - i)) 1

/-- The problem statement -/
theorem fifteen_balls_four_draws :
  orderedDraw 15 4 = 32760 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_balls_four_draws_l742_74292


namespace NUMINAMATH_CALUDE_series_term_equals_original_term_l742_74235

/-- The n-th term of the series -4+7-4+7-4+7-... -/
def seriesTerm (n : ℕ) : ℝ :=
  1.5 + 5.5 * (-1)^n

/-- The original series terms -/
def originalTerm (n : ℕ) : ℝ :=
  if n % 2 = 1 then -4 else 7

theorem series_term_equals_original_term (n : ℕ) :
  seriesTerm n = originalTerm n := by
  sorry

#check series_term_equals_original_term

end NUMINAMATH_CALUDE_series_term_equals_original_term_l742_74235


namespace NUMINAMATH_CALUDE_triangle_properties_l742_74227

/-- Given a triangle ABC with sides a, b, and c, prove the following properties -/
theorem triangle_properties (A B C : ℝ × ℝ) (a b c : ℝ) :
  let AB := B - A
  let BC := C - B
  let CA := A - C
  -- Given condition
  AB • AC + 2 * (-AB) • BC = 3 * (-CA) • (-BC) →
  -- Side lengths
  ‖BC‖ = a ∧ ‖CA‖ = b ∧ ‖AB‖ = c →
  -- Prove these properties
  a^2 + 2*b^2 = 3*c^2 ∧ 
  ∀ (cos_C : ℝ), cos_C = (a^2 + b^2 - c^2) / (2*a*b) → cos_C ≥ Real.sqrt 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l742_74227


namespace NUMINAMATH_CALUDE_marks_team_free_throws_marks_team_free_throws_correct_l742_74271

theorem marks_team_free_throws (marks_two_pointers marks_three_pointers : ℕ) 
  (total_points : ℕ) (h1 : marks_two_pointers = 25) (h2 : marks_three_pointers = 8) 
  (h3 : total_points = 201) : ℕ :=
  let marks_points := 2 * marks_two_pointers + 3 * marks_three_pointers
  let opponents_two_pointers := 2 * marks_two_pointers
  let opponents_three_pointers := marks_three_pointers / 2
  let free_throws := total_points - (marks_points + 2 * opponents_two_pointers + 3 * opponents_three_pointers)
  10

theorem marks_team_free_throws_correct : marks_team_free_throws 25 8 201 rfl rfl rfl = 10 := by
  sorry

end NUMINAMATH_CALUDE_marks_team_free_throws_marks_team_free_throws_correct_l742_74271


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l742_74278

theorem quadratic_two_real_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 4*x + 2 = 0 ∧ a * y^2 - 4*y + 2 = 0) ↔ 
  (a ≤ 2 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l742_74278


namespace NUMINAMATH_CALUDE_marked_cells_bound_l742_74270

/-- Represents a cell color on the board -/
inductive CellColor
| Black
| White

/-- Represents a (2n+1) × (2n+1) board -/
def Board (n : ℕ) := Fin (2*n+1) → Fin (2*n+1) → CellColor

/-- Counts the number of cells of a given color in a row -/
def countInRow (board : Board n) (row : Fin (2*n+1)) (color : CellColor) : ℕ := sorry

/-- Counts the number of cells of a given color in a column -/
def countInColumn (board : Board n) (col : Fin (2*n+1)) (color : CellColor) : ℕ := sorry

/-- Determines if a cell should be marked based on its row -/
def isMarkedInRow (board : Board n) (row col : Fin (2*n+1)) : Bool := sorry

/-- Determines if a cell should be marked based on its column -/
def isMarkedInColumn (board : Board n) (row col : Fin (2*n+1)) : Bool := sorry

/-- Counts the total number of marked cells on the board -/
def countMarkedCells (board : Board n) : ℕ := sorry

/-- Counts the total number of black cells on the board -/
def countBlackCells (board : Board n) : ℕ := sorry

/-- Counts the total number of white cells on the board -/
def countWhiteCells (board : Board n) : ℕ := sorry

/-- The main theorem: The number of marked cells is at least half the minimum of black and white cells -/
theorem marked_cells_bound (n : ℕ) (board : Board n) :
  2 * countMarkedCells board ≥ min (countBlackCells board) (countWhiteCells board) := by
  sorry

end NUMINAMATH_CALUDE_marked_cells_bound_l742_74270


namespace NUMINAMATH_CALUDE_simplify_expression_l742_74272

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l742_74272


namespace NUMINAMATH_CALUDE_nori_crayon_problem_l742_74201

/-- Given the initial number of crayon boxes, crayons per box, crayons given to Mae, and crayons left,
    calculate the difference between crayons given to Lea and Mae. -/
def crayon_difference (boxes : ℕ) (crayons_per_box : ℕ) (given_to_mae : ℕ) (crayons_left : ℕ) : ℕ :=
  boxes * crayons_per_box - given_to_mae - crayons_left - given_to_mae

theorem nori_crayon_problem :
  crayon_difference 4 8 5 15 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nori_crayon_problem_l742_74201


namespace NUMINAMATH_CALUDE_five_letter_words_with_vowels_l742_74226

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels

def word_length : Nat := 5

theorem five_letter_words_with_vowels :
  (alphabet.card ^ word_length) - (consonants.card ^ word_length) = 6752 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_with_vowels_l742_74226


namespace NUMINAMATH_CALUDE_diagonal_intersection_probability_l742_74281

theorem diagonal_intersection_probability (n : ℕ) (h : n > 0) :
  let vertices := 2 * n + 1
  let total_diagonals := (vertices * (vertices - 3)) / 2
  let intersecting_diagonals := vertices.choose 4
  intersecting_diagonals / (total_diagonals.choose 2 : ℚ) = 
    n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_diagonal_intersection_probability_l742_74281


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_l742_74230

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : 1 / a + 2 / b = 1) : 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1 / a' + 2 / b' = 1 → 2 * a + b ≤ 2 * a' + b') ∧ 
  (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 1 / a₀ + 2 / b₀ = 1 ∧ 2 * a₀ + b₀ = 8) := by
  sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_l742_74230


namespace NUMINAMATH_CALUDE_min_domain_for_inverse_l742_74295

-- Define the function g
def g (x : ℝ) : ℝ := (x - 3)^2 + 4

-- State the theorem
theorem min_domain_for_inverse :
  ∃ (d : ℝ), d = 3 ∧ 
  (∀ (d' : ℝ), (∀ (x y : ℝ), x ≥ d' ∧ y ≥ d' ∧ x ≠ y → g x ≠ g y) → d' ≥ d) ∧
  (∀ (x y : ℝ), x ≥ d ∧ y ≥ d ∧ x ≠ y → g x ≠ g y) :=
sorry

end NUMINAMATH_CALUDE_min_domain_for_inverse_l742_74295


namespace NUMINAMATH_CALUDE_carlton_outfits_l742_74214

/-- Represents Carlton's wardrobe and outfit combinations -/
structure Wardrobe where
  button_up_shirts : ℕ
  sweater_vests : ℕ
  outfits : ℕ

/-- Calculates the number of outfits for Carlton -/
def calculate_outfits (w : Wardrobe) : Prop :=
  w.button_up_shirts = 3 ∧
  w.sweater_vests = 2 * w.button_up_shirts ∧
  w.outfits = w.button_up_shirts * w.sweater_vests

/-- Theorem stating that Carlton has 18 outfits -/
theorem carlton_outfits :
  ∃ w : Wardrobe, calculate_outfits w ∧ w.outfits = 18 := by
  sorry


end NUMINAMATH_CALUDE_carlton_outfits_l742_74214


namespace NUMINAMATH_CALUDE_unique_natural_with_square_neighbors_l742_74293

theorem unique_natural_with_square_neighbors :
  ∃! (n : ℕ), ∃ (k m : ℕ), n + 15 = k^2 ∧ n - 14 = m^2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_natural_with_square_neighbors_l742_74293


namespace NUMINAMATH_CALUDE_second_day_speed_l742_74289

/-- Represents the speed and duration of travel for a day -/
structure DayTravel where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (travel : DayTravel) : ℝ := travel.speed * travel.duration

/-- Proves that the speed on the second day of the trip was 6 miles per hour -/
theorem second_day_speed (
  total_distance : ℝ)
  (day1 : DayTravel)
  (day3 : DayTravel)
  (day2_duration1 : ℝ)
  (day2_duration2 : ℝ)
  (h1 : total_distance = 115)
  (h2 : day1.speed = 5 ∧ day1.duration = 7)
  (h3 : day3.speed = 7 ∧ day3.duration = 5)
  (h4 : day2_duration1 = 6)
  (h5 : day2_duration2 = 3)
  : ∃ (day2_speed : ℝ), 
    total_distance = distance day1 + distance day3 + day2_speed * day2_duration1 + (day2_speed / 2) * day2_duration2 ∧ 
    day2_speed = 6 := by
  sorry

end NUMINAMATH_CALUDE_second_day_speed_l742_74289


namespace NUMINAMATH_CALUDE_simplify_expression_l742_74268

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^4 + b^4 = a + b) :
  a / b + b / a - 1 / (a * b^2) = -(a + b) / (a * b^2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l742_74268


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l742_74261

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 0 → x^2 - a*x + a + 3 ≥ 0) → 
  a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l742_74261


namespace NUMINAMATH_CALUDE_mathematicians_set_l742_74216

-- Define the type for famous figures
inductive FamousFigure
| BillGates
| Gauss
| LiuXiang
| Nobel
| ChenJingrun
| ChenXingshen
| Gorky
| Einstein

-- Define the set of all famous figures
def allFigures : Set FamousFigure :=
  {FamousFigure.BillGates, FamousFigure.Gauss, FamousFigure.LiuXiang, 
   FamousFigure.Nobel, FamousFigure.ChenJingrun, FamousFigure.ChenXingshen, 
   FamousFigure.Gorky, FamousFigure.Einstein}

-- Define the property of being a mathematician
def isMathematician : FamousFigure → Prop :=
  fun figure => match figure with
  | FamousFigure.Gauss => True
  | FamousFigure.ChenJingrun => True
  | FamousFigure.ChenXingshen => True
  | _ => False

-- Theorem: The set of mathematicians is equal to {Gauss, Chen Jingrun, Chen Xingshen}
theorem mathematicians_set :
  {figure ∈ allFigures | isMathematician figure} =
  {FamousFigure.Gauss, FamousFigure.ChenJingrun, FamousFigure.ChenXingshen} :=
by sorry

end NUMINAMATH_CALUDE_mathematicians_set_l742_74216


namespace NUMINAMATH_CALUDE_max_subdivision_sides_l742_74252

/-- Represents a convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n > 2

/-- Represents the maximum number of sides in a subdivision polygon -/
def maxSubdivisionSides (n : ℕ) : ℕ := n

/-- Theorem stating that the maximum number of sides in a subdivision polygon is n -/
theorem max_subdivision_sides (n : ℕ) (p : ConvexPolygon n) :
  maxSubdivisionSides n = n := by
  sorry

#eval maxSubdivisionSides 13    -- Should output 13
#eval maxSubdivisionSides 1950  -- Should output 1950

end NUMINAMATH_CALUDE_max_subdivision_sides_l742_74252


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l742_74254

/-- The complex number z = 2i / (1-i) corresponds to a point in the second quadrant of the complex plane. -/
theorem z_in_second_quadrant : ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ (2 * Complex.I) / (1 - Complex.I) = Complex.mk x y := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l742_74254


namespace NUMINAMATH_CALUDE_cricketer_average_score_l742_74204

theorem cricketer_average_score (score1 score2 : ℕ) (matches1 matches2 : ℕ) 
  (h1 : matches1 = 2)
  (h2 : matches2 = 3)
  (h3 : score1 = 60)
  (h4 : score2 = 50) :
  (matches1 * score1 + matches2 * score2) / (matches1 + matches2) = 54 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l742_74204


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l742_74263

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l742_74263


namespace NUMINAMATH_CALUDE_task_completion_time_l742_74236

/-- Given that m men can complete a task in d days, 
    prove that m + r² men will complete the same task in md / (m + r²) days -/
theorem task_completion_time 
  (m d r : ℕ) -- m, d, and r are natural numbers
  (m_pos : 0 < m) -- m is positive
  (d_pos : 0 < d) -- d is positive
  (total_work : ℕ := m * d) -- total work in man-days
  : (↑total_work : ℚ) / (m + r^2 : ℚ) = (↑m * ↑d : ℚ) / (↑m + ↑r^2 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_task_completion_time_l742_74236


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l742_74285

-- Problem 1
theorem problem_1 : 
  (2 + 1/4)^(1/2) - (-0.96)^0 - (3 + 3/8)^(-2/3) + (3/2)^(-2) = 1/2 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  (2 * (a^2)^(1/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = 4*a := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l742_74285


namespace NUMINAMATH_CALUDE_nancy_next_month_games_l742_74246

/-- The number of football games Nancy plans to attend next month -/
def games_next_month (games_this_month games_last_month total_games : ℕ) : ℕ :=
  total_games - (games_this_month + games_last_month)

/-- Proof that Nancy plans to attend 7 games next month -/
theorem nancy_next_month_games :
  games_next_month 9 8 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_next_month_games_l742_74246


namespace NUMINAMATH_CALUDE_cubic_value_given_quadratic_l742_74241

theorem cubic_value_given_quadratic (x : ℝ) : 
  x^2 - 2*x - 1 = 0 → 3*x^3 - 10*x^2 + 5*x + 2027 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_cubic_value_given_quadratic_l742_74241


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l742_74234

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_sum_equals_two : 2 * lg 5 + lg 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l742_74234


namespace NUMINAMATH_CALUDE_product_digit_sum_equals_800_l742_74231

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Represents a number with n repeated digits of 7 -/
def repeated_sevens (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem product_digit_sum_equals_800 :
  sum_of_digits (8 * repeated_sevens 788) = 800 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_equals_800_l742_74231


namespace NUMINAMATH_CALUDE_combined_bus_ride_length_l742_74298

theorem combined_bus_ride_length 
  (vince_ride : ℝ) 
  (zachary_ride : ℝ) 
  (alexandra_ride : ℝ) 
  (h1 : vince_ride = 0.62) 
  (h2 : zachary_ride = 0.5) 
  (h3 : alexandra_ride = 0.72) : 
  vince_ride + zachary_ride + alexandra_ride = 1.84 := by
sorry

end NUMINAMATH_CALUDE_combined_bus_ride_length_l742_74298


namespace NUMINAMATH_CALUDE_simplify_expression_l742_74232

theorem simplify_expression (x y : ℝ) : 3*x + 2*y + 4*x + 5*y + 7 = 7*x + 7*y + 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l742_74232


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_500_l742_74207

theorem fraction_of_powers_equals_500 : (0.5 : ℝ)^4 / (0.05 : ℝ)^3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_500_l742_74207


namespace NUMINAMATH_CALUDE_increase_in_average_marks_l742_74239

/-- Proves that the increase in average marks is 0.5 when a mark is incorrectly entered as 67 instead of 45 in a class of 44 pupils. -/
theorem increase_in_average_marks 
  (num_pupils : ℕ) 
  (wrong_mark : ℕ) 
  (correct_mark : ℕ) 
  (h1 : num_pupils = 44) 
  (h2 : wrong_mark = 67) 
  (h3 : correct_mark = 45) : 
  (wrong_mark - correct_mark : ℚ) / num_pupils = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_average_marks_l742_74239


namespace NUMINAMATH_CALUDE_dvds_sold_per_day_is_497_l742_74291

/-- Represents the DVD business model -/
structure DVDBusiness where
  initialCost : ℕ
  productionCost : ℕ
  sellingPriceFactor : ℚ
  daysPerWeek : ℕ
  totalWeeks : ℕ
  totalProfit : ℕ

/-- Calculates the number of DVDs sold per day -/
def calculateDVDsSoldPerDay (business : DVDBusiness) : ℕ :=
  let sellingPrice := business.productionCost * business.sellingPriceFactor
  let profitPerDVD := sellingPrice - business.productionCost
  let totalDays := business.daysPerWeek * business.totalWeeks
  let profitPerDay := business.totalProfit / totalDays
  (profitPerDay / profitPerDVD).floor.toNat

/-- Theorem stating that the number of DVDs sold per day is 497 -/
theorem dvds_sold_per_day_is_497 (business : DVDBusiness) 
  (h1 : business.initialCost = 2000)
  (h2 : business.productionCost = 6)
  (h3 : business.sellingPriceFactor = 2.5)
  (h4 : business.daysPerWeek = 5)
  (h5 : business.totalWeeks = 20)
  (h6 : business.totalProfit = 448000) :
  calculateDVDsSoldPerDay business = 497 := by
  sorry

#eval calculateDVDsSoldPerDay {
  initialCost := 2000,
  productionCost := 6,
  sellingPriceFactor := 2.5,
  daysPerWeek := 5,
  totalWeeks := 20,
  totalProfit := 448000
}

end NUMINAMATH_CALUDE_dvds_sold_per_day_is_497_l742_74291


namespace NUMINAMATH_CALUDE_magnitude_sum_vectors_l742_74211

/-- Given two planar vectors a and b, prove that |a + 2b| = 2√2 -/
theorem magnitude_sum_vectors (a b : ℝ × ℝ) :
  (a.1 = 2 ∧ a.2 = 0) →  -- a = (2, 0)
  ‖b‖ = 1 →             -- |b| = 1
  a • b = 0 →           -- angle between a and b is 90°
  ‖a + 2 • b‖ = 2 * Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_magnitude_sum_vectors_l742_74211


namespace NUMINAMATH_CALUDE_probability_of_drawing_item_l742_74299

/-- Proves that the probability of drawing each item in a sample is 1/5 given the total number of components and sample size -/
theorem probability_of_drawing_item 
  (total_components : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_components = 100) 
  (h2 : sample_size = 20) : 
  (sample_size : ℚ) / (total_components : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_drawing_item_l742_74299


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l742_74266

theorem geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1/3)
  (h_S : S = 18)
  (h_series : S = a / (1 - r)) :
  a = 12 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l742_74266


namespace NUMINAMATH_CALUDE_tangency_points_x_coordinates_l742_74277

/-- Given a curve y = x^m and a point A(1,0), prove the x-coordinates of the first two tangency points -/
theorem tangency_points_x_coordinates (m : ℕ) (hm : m > 1) :
  let curve (x : ℝ) := x^m
  let tangent_line (a : ℝ) (x : ℝ) := m * a^(m-1) * (x - a) + a^m
  let a₁ := (tangent_line ⁻¹) 0 1  -- x-coordinate where tangent line passes through (1,0)
  let a₂ := (tangent_line ⁻¹) 0 a₁ -- x-coordinate where tangent line passes through (a₁,0)
  a₁ = m / (m - 1) ∧ a₂ = (m / (m - 1))^2 := by
sorry


end NUMINAMATH_CALUDE_tangency_points_x_coordinates_l742_74277


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l742_74243

def A : Set ℝ := {x | x + 2 = 0}
def B : Set ℝ := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : A ∩ B = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l742_74243


namespace NUMINAMATH_CALUDE_a_m_prime_factors_l742_74224

def a_m (m : ℕ) : ℕ := (2^(2*m+1))^2 + 1

def has_at_most_two_prime_factors (n : ℕ) : Prop :=
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ ∀ (r : ℕ), Nat.Prime r → r ∣ n → r = p ∨ r = q

theorem a_m_prime_factors (m : ℕ) :
  has_at_most_two_prime_factors (a_m m) ↔ m = 0 ∨ m = 1 ∨ m = 2 := by sorry

end NUMINAMATH_CALUDE_a_m_prime_factors_l742_74224


namespace NUMINAMATH_CALUDE_system_solution_ratio_l742_74297

/-- Given a system of linear equations with a parameter k, 
    prove that for a specific value of k, the ratio yz/x^2 is constant --/
theorem system_solution_ratio (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  let k : ℝ := 55 / 26
  x + 2 * k * y + 4 * z = 0 ∧
  4 * x + 2 * k * y - 3 * z = 0 ∧
  3 * x + 5 * y - 4 * z = 0 →
  ∃ (c : ℝ), y * z / (x^2) = c :=
by sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l742_74297


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l742_74220

/-- Proves that in a rhombus with one diagonal of 40 m and an area of 600 m², 
    the length of the other diagonal is 30 m. -/
theorem rhombus_diagonal (d₁ d₂ : ℝ) (area : ℝ) : 
  d₁ = 40 → area = 600 → area = (d₁ * d₂) / 2 → d₂ = 30 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l742_74220


namespace NUMINAMATH_CALUDE_min_value_expression_l742_74259

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^2 + 8*a*b + 32*b^2 + 24*b*c + 8*c^2 ≥ 36 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 1 ∧
    a₀^2 + 8*a₀*b₀ + 32*b₀^2 + 24*b₀*c₀ + 8*c₀^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l742_74259


namespace NUMINAMATH_CALUDE_share_difference_l742_74219

/-- Given a distribution ratio and Vasim's share, calculate the difference between Ranjith's and Faruk's shares -/
theorem share_difference (faruk_ratio vasim_ratio ranjith_ratio : ℕ) (vasim_share : ℕ) : 
  faruk_ratio = 3 →
  vasim_ratio = 5 →
  ranjith_ratio = 6 →
  vasim_share = 1500 →
  (ranjith_ratio * vasim_share / vasim_ratio) - (faruk_ratio * vasim_share / vasim_ratio) = 900 := by
sorry

end NUMINAMATH_CALUDE_share_difference_l742_74219


namespace NUMINAMATH_CALUDE_tv_show_episodes_l742_74274

/-- Given a TV show with the following properties:
  * It ran for 10 seasons
  * The first half of seasons had 20 episodes per season
  * There were 225 total episodes
  This theorem proves that the number of episodes per season in the second half was 25. -/
theorem tv_show_episodes (total_seasons : ℕ) (first_half_episodes : ℕ) (total_episodes : ℕ) :
  total_seasons = 10 →
  first_half_episodes = 20 →
  total_episodes = 225 →
  (total_episodes - (total_seasons / 2 * first_half_episodes)) / (total_seasons / 2) = 25 :=
by sorry

end NUMINAMATH_CALUDE_tv_show_episodes_l742_74274


namespace NUMINAMATH_CALUDE_city_fuel_efficiency_l742_74290

/-- Represents the fuel efficiency of a car in miles per gallon -/
structure CarFuelEfficiency where
  highway : ℝ
  city : ℝ

/-- Represents the distance a car can travel on a full tank in miles -/
structure CarRange where
  highway : ℝ
  city : ℝ

/-- The difference between highway and city fuel efficiency in miles per gallon -/
def efficiency_difference : ℝ := 12

theorem city_fuel_efficiency 
  (car_range : CarRange)
  (car_efficiency : CarFuelEfficiency)
  (h1 : car_range.highway = 800)
  (h2 : car_range.city = 500)
  (h3 : car_efficiency.city = car_efficiency.highway - efficiency_difference)
  (h4 : car_range.highway / car_efficiency.highway = car_range.city / car_efficiency.city) :
  car_efficiency.city = 20 := by
sorry

end NUMINAMATH_CALUDE_city_fuel_efficiency_l742_74290
