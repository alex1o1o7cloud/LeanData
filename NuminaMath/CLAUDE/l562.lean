import Mathlib

namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l562_56275

theorem a_gt_one_sufficient_not_necessary_for_a_squared_gt_one :
  (∀ a : ℝ, a > 1 → a^2 > 1) ∧
  (∃ a : ℝ, a^2 > 1 ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l562_56275


namespace NUMINAMATH_CALUDE_sequence_sum_l562_56215

theorem sequence_sum : 
  let a₁ : ℚ := 4/3
  let a₂ : ℚ := 7/5
  let a₃ : ℚ := 11/8
  let a₄ : ℚ := 19/15
  let a₅ : ℚ := 35/27
  let a₆ : ℚ := 67/52
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ - 9 = -17312.5 / 7020 :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_l562_56215


namespace NUMINAMATH_CALUDE_derivative_f_at_2_l562_56273

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem derivative_f_at_2 : 
  deriv f 2 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_2_l562_56273


namespace NUMINAMATH_CALUDE_christopher_stroll_distance_l562_56213

/-- Given Christopher's strolling speed and time, calculate the distance he strolled. -/
theorem christopher_stroll_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (h1 : speed = 4) 
  (h2 : time = 1.25) : 
  speed * time = 5 := by
  sorry

end NUMINAMATH_CALUDE_christopher_stroll_distance_l562_56213


namespace NUMINAMATH_CALUDE_line_properties_l562_56202

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + a^2 * y + 1 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a^2 + 1) * x - b * y + 3 = 0

-- Define parallel lines
def parallel (a b : ℝ) : Prop := ∀ x y, l₁ a x y → l₂ a b x y

-- Define perpendicular lines
def perpendicular (a b : ℝ) : Prop := ∀ x y, l₁ a x y → l₂ a b x y

theorem line_properties (a b : ℝ) :
  (b = -2 ∧ parallel a b → a = 1 ∨ a = -1) ∧
  (perpendicular a b → ∀ c d : ℝ, perpendicular c d → |a * b| ≤ |c * d| ∧ |a * b| = 2) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l562_56202


namespace NUMINAMATH_CALUDE_system_solution_ratio_l562_56235

theorem system_solution_ratio (k x y z : ℚ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + k * y + 4 * z = 0 →
  2 * x + k * y - 3 * z = 0 →
  x + 2 * y - 4 * z = 0 →
  x * z / (y * y) = 59 / 1024 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l562_56235


namespace NUMINAMATH_CALUDE_solution_set_inequality_l562_56205

theorem solution_set_inequality (x : ℝ) : 
  (abs (x - 1) + abs (x - 2) ≥ 5) ↔ (x ≤ -1 ∨ x ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l562_56205


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l562_56256

theorem circle_diameter_ratio (D C : Real) (shaded_ratio : Real) :
  D = 24 →  -- Diameter of circle D
  C < D →   -- Circle C is inside circle D
  shaded_ratio = 7 →  -- Ratio of shaded area to area of circle C
  C = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l562_56256


namespace NUMINAMATH_CALUDE_stating_cube_coloring_theorem_l562_56210

/-- Represents the number of available colors -/
def num_colors : ℕ := 5

/-- Represents the number of faces in a cube -/
def num_faces : ℕ := 6

/-- Represents the number of faces already painted -/
def painted_faces : ℕ := 3

/-- Represents the number of remaining faces to be painted -/
def remaining_faces : ℕ := num_faces - painted_faces

/-- 
  Represents the number of valid coloring schemes for the remaining faces of a cube,
  given that three adjacent faces are already painted with different colors,
  and no two adjacent faces can have the same color.
-/
def valid_coloring_schemes : ℕ := 13

/-- 
  Theorem stating that the number of valid coloring schemes for the remaining faces
  of a cube is equal to 13, given the specified conditions.
-/
theorem cube_coloring_theorem :
  valid_coloring_schemes = 13 :=
sorry

end NUMINAMATH_CALUDE_stating_cube_coloring_theorem_l562_56210


namespace NUMINAMATH_CALUDE_fiftieth_term_is_296_l562_56267

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 50th term of the specific arithmetic sequence -/
def fiftiethTerm : ℝ :=
  arithmeticSequenceTerm 2 6 50

theorem fiftieth_term_is_296 : fiftiethTerm = 296 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_296_l562_56267


namespace NUMINAMATH_CALUDE_jeans_original_cost_l562_56240

/-- The original cost of jeans before discounts -/
def original_cost : ℝ := 49

/-- The summer discount as a percentage -/
def summer_discount : ℝ := 0.5

/-- The additional Wednesday discount in dollars -/
def wednesday_discount : ℝ := 10

/-- The final price after all discounts -/
def final_price : ℝ := 14.5

/-- Theorem stating that the original cost is correct given the discounts and final price -/
theorem jeans_original_cost :
  final_price = original_cost * (1 - summer_discount) - wednesday_discount := by
  sorry


end NUMINAMATH_CALUDE_jeans_original_cost_l562_56240


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l562_56286

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m α) : 
  perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l562_56286


namespace NUMINAMATH_CALUDE_equation_solutions_l562_56277

theorem equation_solutions :
  (∃ x : ℚ, 6 * x - 4 = 3 * x + 2 ∧ x = 2) ∧
  (∃ x : ℚ, x / 4 - 3 / 5 = (x + 1) / 2 ∧ x = -22 / 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l562_56277


namespace NUMINAMATH_CALUDE_least_integer_with_12_factors_l562_56252

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Theorem: 72 is the least positive integer with exactly 12 positive factors -/
theorem least_integer_with_12_factors :
  (∀ m : ℕ+, m < 72 → num_factors m ≠ 12) ∧ num_factors 72 = 12 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_with_12_factors_l562_56252


namespace NUMINAMATH_CALUDE_gcd_power_remainder_l562_56223

theorem gcd_power_remainder (a b : Nat) : 
  (Nat.gcd (2^(30^10) - 2) (2^(30^45) - 2)) % 2013 = 2012 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_remainder_l562_56223


namespace NUMINAMATH_CALUDE_brazil_nut_price_is_five_l562_56225

/-- Represents the price of Brazil nuts per pound -/
def brazil_nut_price : ℝ := 5

/-- Represents the price of cashews per pound -/
def cashew_price : ℝ := 6.75

/-- Represents the total weight of the mixture in pounds -/
def total_mixture_weight : ℝ := 50

/-- Represents the selling price of the mixture per pound -/
def mixture_selling_price : ℝ := 5.70

/-- Represents the weight of cashews used in the mixture in pounds -/
def cashew_weight : ℝ := 20

/-- Theorem stating that the price of Brazil nuts is $5 per pound given the conditions -/
theorem brazil_nut_price_is_five :
  brazil_nut_price = 5 ∧
  cashew_price = 6.75 ∧
  total_mixture_weight = 50 ∧
  mixture_selling_price = 5.70 ∧
  cashew_weight = 20 →
  brazil_nut_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_brazil_nut_price_is_five_l562_56225


namespace NUMINAMATH_CALUDE_mechanic_rate_is_75_l562_56296

/-- Calculates the mechanic's hourly rate given the total work time, part cost, and total amount paid -/
def mechanicHourlyRate (workTime : ℕ) (partCost : ℕ) (totalPaid : ℕ) : ℕ :=
  (totalPaid - partCost) / workTime

/-- Proves that the mechanic's hourly rate is $75 given the problem conditions -/
theorem mechanic_rate_is_75 :
  mechanicHourlyRate 2 150 300 = 75 := by
  sorry

end NUMINAMATH_CALUDE_mechanic_rate_is_75_l562_56296


namespace NUMINAMATH_CALUDE_shaded_area_grid_l562_56211

/-- The area of the shaded region in a grid with specific properties -/
theorem shaded_area_grid (total_width total_height large_triangle_base large_triangle_height small_triangle_base small_triangle_height : ℝ) 
  (hw : total_width = 15)
  (hh : total_height = 5)
  (hlb : large_triangle_base = 15)
  (hlh : large_triangle_height = 3)
  (hsb : small_triangle_base = 3)
  (hsh : small_triangle_height = 4) :
  total_width * total_height - (large_triangle_base * large_triangle_height / 2) + (small_triangle_base * small_triangle_height / 2) = 58.5 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_grid_l562_56211


namespace NUMINAMATH_CALUDE_inverse_of_congruent_area_equal_l562_56274

-- Define the types for triangles and areas
def Triangle : Type := sorry
def Area : Type := sorry

-- Define the congruence relation for triangles
def congruent : Triangle → Triangle → Prop := sorry

-- Define the equality of areas
def area_equal : Area → Area → Prop := sorry

-- Define the area function for triangles
def triangle_area : Triangle → Area := sorry

-- Define the original proposition
def original_proposition : Prop :=
  ∀ (t1 t2 : Triangle), congruent t1 t2 → area_equal (triangle_area t1) (triangle_area t2)

-- Define the inverse proposition
def inverse_proposition : Prop :=
  ∀ (t1 t2 : Triangle), area_equal (triangle_area t1) (triangle_area t2) → congruent t1 t2

-- Theorem stating that the inverse_proposition is the correct inverse of the original_proposition
theorem inverse_of_congruent_area_equal :
  inverse_proposition = (¬original_proposition → ¬(∀ (t1 t2 : Triangle), congruent t1 t2)) := by sorry

end NUMINAMATH_CALUDE_inverse_of_congruent_area_equal_l562_56274


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l562_56201

theorem min_value_expression (a b : ℤ) (h : a > b) :
  (a + 2*b) / (a - b) + (a - b) / (a + 2*b) ≥ 2 :=
sorry

theorem min_value_achievable :
  ∃ (a b : ℤ), a > b ∧ (a + 2*b) / (a - b) + (a - b) / (a + 2*b) = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l562_56201


namespace NUMINAMATH_CALUDE_project_scores_analysis_l562_56289

def scores : List ℝ := [8, 10, 9, 7, 7, 9, 8, 9]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

theorem project_scores_analysis :
  mode scores = 9 ∧
  median scores = 8.5 ∧
  range scores = 3 ∧
  mean scores ≠ 8.4 := by sorry

end NUMINAMATH_CALUDE_project_scores_analysis_l562_56289


namespace NUMINAMATH_CALUDE_square_sum_nonzero_iff_not_both_zero_l562_56279

theorem square_sum_nonzero_iff_not_both_zero (a b : ℝ) :
  a^2 + b^2 ≠ 0 ↔ ¬(a = 0 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_nonzero_iff_not_both_zero_l562_56279


namespace NUMINAMATH_CALUDE_count_numbers_with_three_in_range_l562_56220

def count_numbers_with_three (lower_bound upper_bound : ℕ) : ℕ :=
  sorry

theorem count_numbers_with_three_in_range : 
  count_numbers_with_three 200 499 = 138 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_three_in_range_l562_56220


namespace NUMINAMATH_CALUDE_allan_bought_two_balloons_l562_56212

/-- The number of balloons Allan bought at the park -/
def balloons_bought (allan_initial jake_brought total : ℕ) : ℕ :=
  total - (allan_initial + jake_brought)

/-- Theorem: Allan bought 2 balloons at the park -/
theorem allan_bought_two_balloons : balloons_bought 3 5 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_allan_bought_two_balloons_l562_56212


namespace NUMINAMATH_CALUDE_triangle_inequality_l562_56287

/-- Triangle Inequality Theorem: For any triangle, the sum of the lengths of any two sides
    is greater than the length of the remaining side. -/
theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a + b > c ∧ b + c > a ∧ c + a > b := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l562_56287


namespace NUMINAMATH_CALUDE_max_product_constrained_l562_56259

theorem max_product_constrained (x y : ℕ+) (h : 7 * x + 5 * y = 140) :
  x * y ≤ 140 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_l562_56259


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_at_origin_l562_56288

theorem circle_tangent_to_x_axis_at_origin 
  (D E F : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0 → 
    (∃ r : ℝ, r > 0 ∧ 
      ∀ x y : ℝ, (x^2 + y^2 = r^2) ↔ (x^2 + y^2 + D*x + E*y + F = 0)) ∧
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x| < δ → 
      ∃ y : ℝ, |y| < ε ∧ x^2 + y^2 + D*x + E*y + F = 0) ∧
    (0^2 + 0^2 + D*0 + E*0 + F = 0)) →
  D = 0 ∧ F = 0 ∧ E ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_at_origin_l562_56288


namespace NUMINAMATH_CALUDE_people_remaining_on_bus_l562_56281

/-- The number of people remaining on a bus after a field trip with multiple stops -/
theorem people_remaining_on_bus 
  (left_side : ℕ) 
  (right_side : ℕ) 
  (back_section : ℕ) 
  (standing : ℕ) 
  (teachers : ℕ) 
  (driver : ℕ) 
  (first_stop : ℕ) 
  (second_stop : ℕ) 
  (third_stop : ℕ) 
  (h1 : left_side = 42)
  (h2 : right_side = 38)
  (h3 : back_section = 5)
  (h4 : standing = 15)
  (h5 : teachers = 2)
  (h6 : driver = 1)
  (h7 : first_stop = 15)
  (h8 : second_stop = 19)
  (h9 : third_stop = 5) :
  left_side + right_side + back_section + standing + teachers + driver - 
  (first_stop + second_stop + third_stop) = 64 :=
by sorry

end NUMINAMATH_CALUDE_people_remaining_on_bus_l562_56281


namespace NUMINAMATH_CALUDE_three_numbers_problem_l562_56229

theorem three_numbers_problem (x y z : ℝ) : 
  (x / y = y / z) ∧ 
  (x - (y + z) = 2) ∧ 
  (x + (y - z) / 2 = 9) →
  ((x = 8 ∧ y = 4 ∧ z = 2) ∨ (x = -6.4 ∧ y = 11.2 ∧ z = -19.6)) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l562_56229


namespace NUMINAMATH_CALUDE_square_difference_65_55_l562_56266

theorem square_difference_65_55 : 65^2 - 55^2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_65_55_l562_56266


namespace NUMINAMATH_CALUDE_equation_solution_l562_56219

theorem equation_solution (x : ℝ) :
  (2 * (Real.cos (4 * x) - Real.sin x * Real.cos (3 * x)) = Real.sin (4 * x) + Real.sin (2 * x)) ↔
  (∃ k : ℤ, x = (π / 16) * (4 * ↑k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l562_56219


namespace NUMINAMATH_CALUDE_population_decrease_rate_l562_56291

theorem population_decrease_rate (initial_population : ℕ) (population_after_2_years : ℕ) 
  (h1 : initial_population = 30000)
  (h2 : population_after_2_years = 19200) :
  ∃ (r : ℝ), r = 0.2 ∧ (1 - r)^2 * initial_population = population_after_2_years :=
by sorry

end NUMINAMATH_CALUDE_population_decrease_rate_l562_56291


namespace NUMINAMATH_CALUDE_part_one_part_two_l562_56222

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 4| + |x - a|

-- Part I
theorem part_one : 
  {x : ℝ | f 2 x > 10} = {x : ℝ | x > 8 ∨ x < -2} :=
sorry

-- Part II
theorem part_two : 
  (∀ x : ℝ, f a x ≥ 1) → (a ≥ 5 ∨ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l562_56222


namespace NUMINAMATH_CALUDE_cosine_A_is_half_area_of_triangle_l562_56249

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  t.c * Real.cos t.A + t.a * Real.cos t.C = 2 * t.b * Real.cos t.A

def satisfiesExtraConditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 7 ∧ t.b + t.c = 4

-- Theorem 1
theorem cosine_A_is_half (t : Triangle) (h : satisfiesCondition t) : 
  Real.cos t.A = 1/2 := by sorry

-- Theorem 2
theorem area_of_triangle (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : satisfiesExtraConditions t) : 
  (1/2 * t.b * t.c * Real.sin t.A) = (3 * Real.sqrt 3) / 4 := by sorry

end NUMINAMATH_CALUDE_cosine_A_is_half_area_of_triangle_l562_56249


namespace NUMINAMATH_CALUDE_sunday_newspaper_delivery_l562_56261

theorem sunday_newspaper_delivery (total : ℕ) (difference : ℕ) 
  (h1 : total = 110)
  (h2 : difference = 20) :
  ∃ (saturday sunday : ℕ), 
    saturday + sunday = total ∧ 
    sunday = saturday + difference ∧ 
    sunday = 65 := by
  sorry

end NUMINAMATH_CALUDE_sunday_newspaper_delivery_l562_56261


namespace NUMINAMATH_CALUDE_rain_and_humidity_probability_l562_56264

/-- The probability of rain in a coastal city in Zhejiang -/
def prob_rain : ℝ := 0.4

/-- The probability that the humidity exceeds 70% on rainy days -/
def prob_humidity_given_rain : ℝ := 0.6

/-- The probability that it rains and the humidity exceeds 70% -/
def prob_rain_and_humidity : ℝ := prob_rain * prob_humidity_given_rain

theorem rain_and_humidity_probability :
  prob_rain_and_humidity = 0.24 :=
sorry

end NUMINAMATH_CALUDE_rain_and_humidity_probability_l562_56264


namespace NUMINAMATH_CALUDE_intersection_of_sets_l562_56284

theorem intersection_of_sets (a : ℝ) : 
  let A : Set ℝ := {-a, a^2, a^2 + a}
  let B : Set ℝ := {-1, -1 - a, 1 + a^2}
  (A ∩ B).Nonempty → A ∩ B = {-1, 2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l562_56284


namespace NUMINAMATH_CALUDE_margo_walk_distance_l562_56248

/-- Calculates the total distance of a round trip given the times for each leg and the average speed -/
def round_trip_distance (outbound_time inbound_time : ℚ) (average_speed : ℚ) : ℚ :=
  let total_time := outbound_time + inbound_time
  average_speed * (total_time / 60)

/-- Proves that given the specific conditions of Margo's walk, the total distance is 2 miles -/
theorem margo_walk_distance :
  round_trip_distance (15 : ℚ) (25 : ℚ) (3 : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_margo_walk_distance_l562_56248


namespace NUMINAMATH_CALUDE_magic_square_sum_div_by_3_l562_56209

/-- Definition of a 3x3 magic square -/
def is_magic_square (a : Fin 9 → ℕ) (S : ℕ) : Prop :=
  -- Row sums
  (a 0 + a 1 + a 2 = S) ∧
  (a 3 + a 4 + a 5 = S) ∧
  (a 6 + a 7 + a 8 = S) ∧
  -- Column sums
  (a 0 + a 3 + a 6 = S) ∧
  (a 1 + a 4 + a 7 = S) ∧
  (a 2 + a 5 + a 8 = S) ∧
  -- Diagonal sums
  (a 0 + a 4 + a 8 = S) ∧
  (a 2 + a 4 + a 6 = S)

/-- Theorem: The sum of a third-order magic square is divisible by 3 -/
theorem magic_square_sum_div_by_3 (a : Fin 9 → ℕ) (S : ℕ) 
  (h : is_magic_square a S) : 
  3 ∣ S :=
by sorry

end NUMINAMATH_CALUDE_magic_square_sum_div_by_3_l562_56209


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l562_56257

theorem arithmetic_calculations :
  ((-15) - (-5) + 6 = -4) ∧
  (81 / (-9/5) * (5/9) = -25) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l562_56257


namespace NUMINAMATH_CALUDE_sugar_for_muffins_l562_56200

/-- Given a recipe for muffins, calculate the required sugar for a larger batch -/
theorem sugar_for_muffins (original_muffins original_sugar target_muffins : ℕ) :
  original_muffins > 0 →
  original_sugar > 0 →
  target_muffins > 0 →
  (original_sugar * target_muffins) / original_muffins = 
    (3 * 72) / 24 :=
by
  sorry

#eval (3 * 72) / 24  -- This should output 9

end NUMINAMATH_CALUDE_sugar_for_muffins_l562_56200


namespace NUMINAMATH_CALUDE_math_quiz_items_l562_56295

theorem math_quiz_items (score_percentage : ℚ) (num_mistakes : ℕ) : 
  score_percentage = 80 → num_mistakes = 5 → 
  (100 : ℚ) * num_mistakes / (100 - score_percentage) = 25 :=
by sorry

end NUMINAMATH_CALUDE_math_quiz_items_l562_56295


namespace NUMINAMATH_CALUDE_smallest_value_3a_plus_1_l562_56203

theorem smallest_value_3a_plus_1 (a : ℂ) (h : 8 * a^2 + 6 * a + 2 = 0) :
  ∃ (z : ℂ), 3 * z + 1 = -1/8 ∧ ∀ (w : ℂ), 8 * w^2 + 6 * w + 2 = 0 → Complex.re (3 * w + 1) ≥ -1/8 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_3a_plus_1_l562_56203


namespace NUMINAMATH_CALUDE_range_of_x_minus_sqrt3y_l562_56258

theorem range_of_x_minus_sqrt3y (x y : ℝ) 
  (h : x^2 + y^2 - 2*x + 2*Real.sqrt 3*y + 3 = 0) :
  ∃ (min max : ℝ), min = 2 ∧ max = 6 ∧ 
    (∀ z, z = x - Real.sqrt 3 * y → min ≤ z ∧ z ≤ max) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_minus_sqrt3y_l562_56258


namespace NUMINAMATH_CALUDE_quadratic_factorization_l562_56242

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 56 * x + 49 = (4 * x - 7)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l562_56242


namespace NUMINAMATH_CALUDE_ratio_equality_l562_56243

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (x - z) = (x + y) / z ∧ (x + y) / z = x / y) : 
  x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l562_56243


namespace NUMINAMATH_CALUDE_distance_calculation_l562_56221

theorem distance_calculation (speed : ℝ) (time : ℝ) (h1 : speed = 100) (h2 : time = 5) :
  speed * time = 500 := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l562_56221


namespace NUMINAMATH_CALUDE_hcf_problem_l562_56285

/-- Given two positive integers with specific LCM and maximum value properties, prove their HCF is 4 -/
theorem hcf_problem (a b : ℕ+) : 
  (∃ (lcm : ℕ+), Nat.lcm a b = lcm ∧ ∃ (hcf : ℕ+), lcm = hcf * 10 * 20) →
  (max a b = 840) →
  Nat.gcd a b = 4 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l562_56285


namespace NUMINAMATH_CALUDE_sum_of_primes_with_square_property_l562_56292

theorem sum_of_primes_with_square_property : ∃ (S : Finset Nat),
  (∀ p ∈ S, Nat.Prime p ∧ ∃ q, Nat.Prime q ∧ ∃ k, p^2 + p*q + q^2 = k^2) ∧
  (∀ p, Nat.Prime p → (∃ q, Nat.Prime q ∧ ∃ k, p^2 + p*q + q^2 = k^2) → p ∈ S) ∧
  S.sum id = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_primes_with_square_property_l562_56292


namespace NUMINAMATH_CALUDE_discount_order_matters_l562_56232

/-- Proves that applying a percentage discount followed by a fixed discount
    results in a lower final price than the reverse order. -/
theorem discount_order_matters (initial_price percent_off fixed_off : ℚ) 
  (h1 : initial_price > 0)
  (h2 : percent_off > 0 ∧ percent_off < 1)
  (h3 : fixed_off > 0)
  (h4 : fixed_off < initial_price) :
  (1 - percent_off) * initial_price - fixed_off < initial_price - fixed_off - percent_off * (initial_price - fixed_off) :=
by sorry

end NUMINAMATH_CALUDE_discount_order_matters_l562_56232


namespace NUMINAMATH_CALUDE_dave_guitar_strings_l562_56226

/-- The number of guitar strings Dave breaks per night -/
def strings_per_night : ℕ := 2

/-- The number of shows Dave performs per week -/
def shows_per_week : ℕ := 6

/-- The number of weeks Dave performs -/
def total_weeks : ℕ := 12

/-- The total number of guitar strings Dave needs to replace -/
def total_strings_needed : ℕ := 144

/-- Theorem stating that the total number of guitar strings Dave needs to replace is 144 -/
theorem dave_guitar_strings :
  strings_per_night * shows_per_week * total_weeks = total_strings_needed :=
by sorry

end NUMINAMATH_CALUDE_dave_guitar_strings_l562_56226


namespace NUMINAMATH_CALUDE_negation_equivalence_l562_56250

/-- The proposition that there exists a real number 'a' for which the equation ax^2 + 1 = 0 has a real solution. -/
def original_proposition : Prop :=
  ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0

/-- The negation of the original proposition. -/
def negation_proposition : Prop :=
  ∀ a : ℝ, ∀ x : ℝ, a * x^2 + 1 ≠ 0

/-- Theorem stating that the negation of the original proposition is equivalent to the negation_proposition. -/
theorem negation_equivalence : ¬original_proposition ↔ negation_proposition :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l562_56250


namespace NUMINAMATH_CALUDE_decimal_to_base_five_l562_56265

theorem decimal_to_base_five : 
  (2 * 5^3 + 0 * 5^2 + 1 * 5^1 + 1 * 5^0 : ℕ) = 256 := by sorry

end NUMINAMATH_CALUDE_decimal_to_base_five_l562_56265


namespace NUMINAMATH_CALUDE_bus_ride_difference_l562_56290

def tess_to_noah : ℝ := 0.75
def tess_noah_to_kayla : ℝ := 0.85
def tess_kayla_to_school : ℝ := 1.15

def oscar_to_charlie : ℝ := 0.25
def oscar_charlie_to_school : ℝ := 1.35

theorem bus_ride_difference : 
  (tess_to_noah + tess_noah_to_kayla + tess_kayla_to_school) - 
  (oscar_to_charlie + oscar_charlie_to_school) = 1.15 := by
  sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l562_56290


namespace NUMINAMATH_CALUDE_unchanged_flipped_nine_digit_numbers_l562_56224

/-- 
Given that:
- A 9-digit number is considered unchanged when flipped if it reads the same upside down.
- Digits 0, 1, and 8 remain unchanged when flipped.
- Digits 6 and 9 become each other when flipped.
- Other digits have no meaning when flipped.

This theorem states that the number of 9-digit numbers that remain unchanged when flipped is 1500.
-/
theorem unchanged_flipped_nine_digit_numbers : ℕ := by
  -- Define the set of digits that remain unchanged when flipped
  let unchanged_digits : Finset ℕ := {0, 1, 8}
  
  -- Define the set of digit pairs that become each other when flipped
  let swapped_digits : Finset (ℕ × ℕ) := {(6, 9), (9, 6)}
  
  -- Define the number of valid options for the first and last digit
  let first_last_options : ℕ := 4
  
  -- Define the number of valid options for the second, third, fourth, and eighth digit
  let middle_pair_options : ℕ := 5
  
  -- Define the number of valid options for the center digit
  let center_options : ℕ := 3
  
  -- Calculate the total number of valid 9-digit numbers
  let total : ℕ := first_last_options * middle_pair_options^3 * center_options
  
  -- Assert that the total is equal to 1500
  have h : total = 1500 := by sorry
  
  -- Return the result
  exact 1500


end NUMINAMATH_CALUDE_unchanged_flipped_nine_digit_numbers_l562_56224


namespace NUMINAMATH_CALUDE_not_or_implies_both_false_l562_56255

theorem not_or_implies_both_false (p q : Prop) : 
  ¬(p ∨ q) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_or_implies_both_false_l562_56255


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l562_56272

theorem sum_of_powers_of_i_is_zero : Complex.I + Complex.I^2 + Complex.I^3 + Complex.I^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l562_56272


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l562_56247

theorem largest_n_satisfying_inequality : 
  ∃ (n : ℕ), (n ≤ 6 ∧ ((1 : ℚ) / 5 + (n : ℚ) / 8 + 1 < 2)) ∧
  ∀ (m : ℕ), m > 6 → ((1 : ℚ) / 5 + (m : ℚ) / 8 + 1 ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l562_56247


namespace NUMINAMATH_CALUDE_df_length_l562_56214

/-- Right triangle ABC with square ABDE and angle bisector intersection -/
structure RightTriangleWithSquare where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  -- Conditions
  right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0
  ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 21
  bc_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 28
  square_abde : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0 ∧
                (E.1 - B.1) * (D.1 - B.1) + (E.2 - B.2) * (D.2 - B.2) = 0 ∧
                Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  f_on_de : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (t * D.1 + (1 - t) * E.1, t * D.2 + (1 - t) * E.2)
  f_on_bisector : ∃ s : ℝ, s > 0 ∧ F = (C.1 + s * (A.1 + B.1 - 2 * C.1), C.2 + s * (A.2 + B.2 - 2 * C.2))

/-- The length of DF is 15 -/
theorem df_length (t : RightTriangleWithSquare) : 
  Real.sqrt ((t.D.1 - t.F.1)^2 + (t.D.2 - t.F.2)^2) = 15 := by
  sorry


end NUMINAMATH_CALUDE_df_length_l562_56214


namespace NUMINAMATH_CALUDE_geese_count_l562_56280

/-- Given a marsh with ducks and geese, calculate the number of geese -/
theorem geese_count (total_birds ducks : ℕ) (h1 : total_birds = 95) (h2 : ducks = 37) :
  total_birds - ducks = 58 := by
  sorry

end NUMINAMATH_CALUDE_geese_count_l562_56280


namespace NUMINAMATH_CALUDE_eve_envelope_count_l562_56227

def envelope_numbers : List ℕ := [1, 2, 4, 8, 16, 32, 64, 128]

theorem eve_envelope_count :
  ∀ (eve_numbers alie_numbers : List ℕ),
    eve_numbers ++ alie_numbers = envelope_numbers →
    eve_numbers.sum = alie_numbers.sum + 31 →
    eve_numbers.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_eve_envelope_count_l562_56227


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l562_56294

/-- Calculates the number of overtime hours worked given the regular pay rate,
    regular hours limit, and total pay received. -/
def overtime_hours (regular_rate : ℚ) (regular_hours_limit : ℕ) (total_pay : ℚ) : ℚ :=
  let regular_pay := regular_rate * regular_hours_limit
  let overtime_rate := 2 * regular_rate
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate

/-- Proves that given the specified conditions, the number of overtime hours is 12. -/
theorem overtime_hours_calculation :
  let regular_rate : ℚ := 3
  let regular_hours_limit : ℕ := 40
  let total_pay : ℚ := 192
  overtime_hours regular_rate regular_hours_limit total_pay = 12 := by
  sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l562_56294


namespace NUMINAMATH_CALUDE_determine_set_B_l562_56282

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define the theorem
theorem determine_set_B (A B : Set Nat) 
  (h1 : (A ∪ B)ᶜ = {1}) 
  (h2 : A ∩ Bᶜ = {3}) 
  (h3 : A ⊆ U) 
  (h4 : B ⊆ U) : 
  B = {2, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_determine_set_B_l562_56282


namespace NUMINAMATH_CALUDE_parabola_coefficient_b_l562_56260

/-- Given a parabola y = ax^2 + bx + c with vertex at (h, -h) and y-intercept at (0, h),
    where h ≠ 0, the coefficient b equals -4. -/
theorem parabola_coefficient_b (a b c h : ℝ) : 
  h ≠ 0 →
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 - h) →
  c = h →
  b = -4 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_b_l562_56260


namespace NUMINAMATH_CALUDE_improved_milk_production_l562_56216

/-- Given initial milk production parameters and an efficiency increase,
    calculate the new milk production for a different number of cows and days. -/
theorem improved_milk_production
  (a b c d e : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0)
  (h_initial : b / (a * c) = initial_rate)
  (h_efficiency_increase : new_rate = initial_rate * 1.2) :
  new_rate * d * e = (1.2 * b * d * e) / (a * c) :=
sorry

end NUMINAMATH_CALUDE_improved_milk_production_l562_56216


namespace NUMINAMATH_CALUDE_students_liking_both_pizza_and_burgers_l562_56298

theorem students_liking_both_pizza_and_burgers 
  (total : ℕ) 
  (pizza : ℕ) 
  (burgers : ℕ) 
  (neither : ℕ) 
  (h1 : total = 50) 
  (h2 : pizza = 22) 
  (h3 : burgers = 20) 
  (h4 : neither = 14) : 
  pizza + burgers - (total - neither) = 6 := by
sorry

end NUMINAMATH_CALUDE_students_liking_both_pizza_and_burgers_l562_56298


namespace NUMINAMATH_CALUDE_polynomial_factorization_l562_56270

theorem polynomial_factorization (z : ℂ) : z^6 - 64*z^2 = z^2 * (z^2 - 8) * (z^2 + 8) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l562_56270


namespace NUMINAMATH_CALUDE_northton_capsule_depth_l562_56237

/-- The depth of Northton's time capsule given Southton's depth and the relationship between them. -/
theorem northton_capsule_depth (southton_depth : ℕ) (h1 : southton_depth = 15) :
  southton_depth * 4 - 12 = 48 := by
  sorry

end NUMINAMATH_CALUDE_northton_capsule_depth_l562_56237


namespace NUMINAMATH_CALUDE_first_floor_bedrooms_l562_56268

theorem first_floor_bedrooms 
  (total : ℕ) 
  (second_floor : ℕ) 
  (third_floor : ℕ) 
  (fourth_floor : ℕ) 
  (h1 : total = 22) 
  (h2 : second_floor = 6) 
  (h3 : third_floor = 4) 
  (h4 : fourth_floor = 3) : 
  total - (second_floor + third_floor + fourth_floor) = 9 := by
  sorry

end NUMINAMATH_CALUDE_first_floor_bedrooms_l562_56268


namespace NUMINAMATH_CALUDE_pet_food_discount_l562_56278

/-- Proves that the regular discount is 30% given the conditions of the problem -/
theorem pet_food_discount (msrp : ℝ) (sale_price : ℝ) (additional_discount : ℝ) :
  msrp = 45 →
  sale_price = 25.2 →
  additional_discount = 20 →
  ∃ (regular_discount : ℝ),
    sale_price = msrp * (1 - regular_discount / 100) * (1 - additional_discount / 100) ∧
    regular_discount = 30 := by
  sorry

#check pet_food_discount

end NUMINAMATH_CALUDE_pet_food_discount_l562_56278


namespace NUMINAMATH_CALUDE_division_into_proportional_parts_l562_56218

theorem division_into_proportional_parts :
  let total : ℚ := 156
  let proportions : List ℚ := [2, 1/2, 1/4, 1/8]
  let parts := proportions.map (λ p => p * (total / proportions.sum))
  parts[2] = 13 + 15/23 := by sorry

end NUMINAMATH_CALUDE_division_into_proportional_parts_l562_56218


namespace NUMINAMATH_CALUDE_balls_sold_count_l562_56251

def selling_price : ℕ := 720
def cost_price_per_ball : ℕ := 72
def loss : ℕ := 5 * cost_price_per_ball

theorem balls_sold_count :
  ∃ n : ℕ, n * cost_price_per_ball - selling_price = loss ∧ n = 15 :=
by sorry

end NUMINAMATH_CALUDE_balls_sold_count_l562_56251


namespace NUMINAMATH_CALUDE_strawberry_plants_l562_56231

theorem strawberry_plants (initial_plants : ℕ) : 
  (initial_plants * 2 * 3 * 4 - 4 = 500) → initial_plants = 21 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_plants_l562_56231


namespace NUMINAMATH_CALUDE_inequality_theorem_l562_56254

theorem inequality_theorem (a b c : ℝ) (θ : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_ineq : a * (Real.cos θ)^2 + b * (Real.sin θ)^2 < c) :
  Real.sqrt a * (Real.cos θ)^2 + Real.sqrt b * (Real.sin θ)^2 < Real.sqrt c := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l562_56254


namespace NUMINAMATH_CALUDE_set_intersection_proof_l562_56238

def A : Set ℝ := {x : ℝ | |2*x - 1| < 6}
def B : Set ℝ := {-3, 0, 1, 2, 3, 4}

theorem set_intersection_proof : A ∩ B = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_proof_l562_56238


namespace NUMINAMATH_CALUDE_some_number_value_l562_56204

theorem some_number_value : ∃ (some_number : ℝ), 
  |5 - 8 * (3 - some_number)| - |5 - 11| = 71 ∧ some_number = 12 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l562_56204


namespace NUMINAMATH_CALUDE_driving_time_proof_l562_56293

/-- Proves that given the conditions of the driving problem, the driving times for route one and route two are 2 hours and 2.5 hours respectively. -/
theorem driving_time_proof (distance_one : ℝ) (distance_two : ℝ) (time_diff : ℝ) (speed_ratio : ℝ) :
  distance_one = 180 →
  distance_two = 150 →
  time_diff = 0.5 →
  speed_ratio = 1.5 →
  ∃ (time_one time_two : ℝ),
    time_one = 2 ∧
    time_two = 2.5 ∧
    time_two = time_one + time_diff ∧
    distance_one / time_one = speed_ratio * (distance_two / time_two) :=
by sorry


end NUMINAMATH_CALUDE_driving_time_proof_l562_56293


namespace NUMINAMATH_CALUDE_embankment_construction_time_l562_56234

theorem embankment_construction_time 
  (workers_initial : ℕ) 
  (days_initial : ℕ) 
  (embankments : ℕ) 
  (workers_new : ℕ) 
  (h1 : workers_initial = 75)
  (h2 : days_initial = 4)
  (h3 : embankments = 2)
  (h4 : workers_new = 60) :
  ∃ (days_new : ℕ), 
    (workers_initial * days_initial = workers_new * days_new) ∧ 
    (days_new = 5) := by
  sorry

end NUMINAMATH_CALUDE_embankment_construction_time_l562_56234


namespace NUMINAMATH_CALUDE_circle_and_chord_theorem_l562_56207

/-- The polar coordinate equation of a circle C that passes through the point (√2, π/4)
    and has its center at the intersection of the polar axis and the line ρ sin(θ - π/3) = -√3/2 -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The length of the chord intercepted by the line θ = π/3 on the circle C defined by ρ = 2cos(θ) -/
def chord_length : ℝ := 1

theorem circle_and_chord_theorem :
  /- Circle C passes through (√2, π/4) -/
  (circle_equation (Real.sqrt 2) (π / 4)) ∧
  /- The center of C is at the intersection of the polar axis and ρ sin(θ - π/3) = -√3/2 -/
  (∃ ρ₀ : ℝ, ρ₀ * Real.sin (0 - π / 3) = -Real.sqrt 3 / 2) ∧
  /- The polar coordinate equation of circle C is ρ = 2cos(θ) -/
  (∀ ρ θ : ℝ, circle_equation ρ θ ↔ ρ = 2 * Real.cos θ) ∧
  /- The length of the chord intercepted by θ = π/3 on circle C is 1 -/
  chord_length = 1 := by
    sorry

end NUMINAMATH_CALUDE_circle_and_chord_theorem_l562_56207


namespace NUMINAMATH_CALUDE_min_box_height_is_seven_l562_56253

/-- Represents the side length of the square base of the box -/
def base_side : ℝ → ℝ := λ x => x

/-- Represents the height of the box -/
def box_height : ℝ → ℝ := λ x => x + 5

/-- Calculates the surface area of the box -/
def surface_area : ℝ → ℝ := λ x => 2 * x^2 + 4 * x * (x + 5)

/-- Theorem stating that the minimum height of the box satisfying the conditions is 7 -/
theorem min_box_height_is_seven :
  ∃ x : ℝ, x > 0 ∧ 
    surface_area x ≥ 120 ∧
    box_height x = 7 ∧
    ∀ y : ℝ, y > 0 ∧ surface_area y ≥ 120 → box_height y ≥ box_height x :=
by
  sorry


end NUMINAMATH_CALUDE_min_box_height_is_seven_l562_56253


namespace NUMINAMATH_CALUDE_largest_odd_integer_in_range_l562_56246

theorem largest_odd_integer_in_range : 
  ∃ (x : ℤ), (x % 2 = 1) ∧ (1/4 < x/6) ∧ (x/6 < 7/9) ∧
  ∀ (y : ℤ), (y % 2 = 1) ∧ (1/4 < y/6) ∧ (y/6 < 7/9) → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_odd_integer_in_range_l562_56246


namespace NUMINAMATH_CALUDE_inequality_solution_l562_56245

theorem inequality_solution (x : ℝ) :
  (3 * (x + 2) - 1 ≥ 5 - 2 * (x - 2)) ↔ (x ≥ 4 / 5) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l562_56245


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l562_56297

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 2 > 0}

-- State the theorem
theorem intersection_of_M_and_N : 
  M ∩ N = {x : ℝ | (-4 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 7)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l562_56297


namespace NUMINAMATH_CALUDE_angle_ratios_l562_56271

theorem angle_ratios (α : Real) (h : Real.tan α = -3/4) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -3/4 ∧
  2 + Real.sin α * Real.cos α - (Real.cos α)^2 = 22/25 := by
  sorry

end NUMINAMATH_CALUDE_angle_ratios_l562_56271


namespace NUMINAMATH_CALUDE_simplify_fraction_l562_56228

theorem simplify_fraction : (5^6 + 5^3) / (5^5 - 5^2) = 315 / 62 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l562_56228


namespace NUMINAMATH_CALUDE_josh_marbles_l562_56276

/-- Theorem: If Josh had 16 marbles and lost 7 marbles, he now has 9 marbles. -/
theorem josh_marbles (initial : ℕ) (lost : ℕ) (final : ℕ) 
  (h1 : initial = 16) 
  (h2 : lost = 7) 
  (h3 : final = initial - lost) : 
  final = 9 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l562_56276


namespace NUMINAMATH_CALUDE_smallest_transformed_sum_l562_56230

/-- The number of faces on a standard die -/
def standardDieFaces : ℕ := 6

/-- The sum we want to compare with -/
def targetSum : ℕ := 980

/-- A function to calculate the transformed sum given the number of dice -/
def transformedSum (n : ℕ) : ℤ := 5 * n - targetSum

/-- The proposition that proves the smallest possible value of S -/
theorem smallest_transformed_sum :
  ∃ (n : ℕ), 
    (n * standardDieFaces ≥ targetSum) ∧ 
    (∀ m : ℕ, m < n → m * standardDieFaces < targetSum) ∧
    (transformedSum n = 5) ∧
    (∀ k : ℕ, k < n → transformedSum k < 5) := by
  sorry

end NUMINAMATH_CALUDE_smallest_transformed_sum_l562_56230


namespace NUMINAMATH_CALUDE_expected_heads_is_60_l562_56283

/-- The number of coins -/
def num_coins : ℕ := 64

/-- The probability of getting heads on a single flip -/
def p_heads : ℚ := 1/2

/-- The number of possible tosses for each coin -/
def max_tosses : ℕ := 4

/-- The probability of a coin showing heads after up to four tosses -/
def prob_heads_after_four : ℚ :=
  p_heads + (1 - p_heads) * p_heads + 
  (1 - p_heads)^2 * p_heads + (1 - p_heads)^3 * p_heads

/-- The expected number of coins showing heads after the series of tosses -/
def expected_heads : ℚ := num_coins * prob_heads_after_four

theorem expected_heads_is_60 : expected_heads = 60 := by
  sorry

end NUMINAMATH_CALUDE_expected_heads_is_60_l562_56283


namespace NUMINAMATH_CALUDE_sum_of_circle_circumferences_l562_56239

/-- The sum of circumferences of an infinite series of circles inscribed in an equilateral triangle -/
theorem sum_of_circle_circumferences (r : ℝ) (h : r = 1) : 
  (2 * π * r) + (3 * (2 * π * r * (∑' n, (1/3)^n))) = 5 * π :=
sorry

end NUMINAMATH_CALUDE_sum_of_circle_circumferences_l562_56239


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_theorem_l562_56206

/-- Represents a rectangular parallelepiped -/
structure RectParallelepiped where
  base_side : ℝ
  cos_angle : ℝ

/-- Represents a vector configuration -/
structure VectorConfig where
  a_magnitude : ℝ
  a_dot_e : ℝ

theorem rectangular_parallelepiped_theorem (rp : RectParallelepiped) (vc : VectorConfig) :
  rp.base_side = 2 * Real.sqrt 2 →
  rp.cos_angle = Real.sqrt 3 / 3 →
  vc.a_magnitude = 2 * Real.sqrt 6 →
  vc.a_dot_e = 2 * Real.sqrt 2 →
  (∃ (sphere_surface_area : ℝ), sphere_surface_area = 24 * Real.pi) ∧
  (∃ (min_value : ℝ), min_value = 2 * Real.sqrt 2) := by
  sorry

#check rectangular_parallelepiped_theorem

end NUMINAMATH_CALUDE_rectangular_parallelepiped_theorem_l562_56206


namespace NUMINAMATH_CALUDE_max_m_plus_2n_max_fraction_min_fraction_l562_56244

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define a point M on the circle C
def M (m n : ℝ) : Prop := C m n

-- Theorem for the maximum value of m + 2n
theorem max_m_plus_2n :
  ∃ (max : ℝ), (∀ m n, M m n → m + 2*n ≤ max) ∧ (∃ m n, M m n ∧ m + 2*n = max) ∧ max = 16 + 2*Real.sqrt 10 :=
sorry

-- Theorem for the maximum value of (n-3)/(m+2)
theorem max_fraction :
  ∃ (max : ℝ), (∀ m n, M m n → (n - 3) / (m + 2) ≤ max) ∧ (∃ m n, M m n ∧ (n - 3) / (m + 2) = max) ∧ max = 2 + Real.sqrt 3 :=
sorry

-- Theorem for the minimum value of (n-3)/(m+2)
theorem min_fraction :
  ∃ (min : ℝ), (∀ m n, M m n → min ≤ (n - 3) / (m + 2)) ∧ (∃ m n, M m n ∧ (n - 3) / (m + 2) = min) ∧ min = 2 - Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_m_plus_2n_max_fraction_min_fraction_l562_56244


namespace NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l562_56241

/-- The piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2 * x + a else x + 1

/-- The theorem stating the range of a for which f is monotonic -/
theorem range_of_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l562_56241


namespace NUMINAMATH_CALUDE_height_difference_approx_10_inches_l562_56269

-- Define constants
def mark_height_cm : ℝ := 160
def mike_height_cm : ℝ := 185
def cm_to_m : ℝ := 0.01
def m_to_ft : ℝ := 3.28084
def ft_to_in : ℝ := 12

-- Define the height difference function
def height_difference_inches (h1 h2 : ℝ) : ℝ :=
  (h2 - h1) * cm_to_m * m_to_ft * ft_to_in

-- Theorem statement
theorem height_difference_approx_10_inches :
  ∃ ε > 0, abs (height_difference_inches mark_height_cm mike_height_cm - 10) < ε :=
sorry

end NUMINAMATH_CALUDE_height_difference_approx_10_inches_l562_56269


namespace NUMINAMATH_CALUDE_zero_point_implies_a_range_l562_56236

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 2

-- Define the theorem
theorem zero_point_implies_a_range (a : ℝ) : 
  (∃ x ∈ Set.Icc (-2) 1, g a x = 0) → 
  (a < 2) → 
  a ∈ Set.Icc (-3/2) 2 := by
sorry

end NUMINAMATH_CALUDE_zero_point_implies_a_range_l562_56236


namespace NUMINAMATH_CALUDE_probability_purple_marble_l562_56263

theorem probability_purple_marble (blue_prob green_prob : ℝ) 
  (h1 : blue_prob = 0.3)
  (h2 : green_prob = 0.4)
  (h3 : ∃ purple_prob : ℝ, blue_prob + green_prob + purple_prob = 1) :
  ∃ purple_prob : ℝ, purple_prob = 0.3 ∧ blue_prob + green_prob + purple_prob = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_purple_marble_l562_56263


namespace NUMINAMATH_CALUDE_tangent_intersection_y_coordinate_l562_56299

/-- 
Given a parabola y = x^2 + 1 and two points A and B on it with perpendicular tangents,
this theorem states that the y-coordinate of the intersection point P of these tangents is 3/4.
-/
theorem tangent_intersection_y_coordinate (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + 1
  let A : ℝ × ℝ := (a, f a)
  let B : ℝ × ℝ := (b, f b)
  let tangent_A : ℝ → ℝ := λ x => 2*a*x - a^2 + 1
  let tangent_B : ℝ → ℝ := λ x => 2*b*x - b^2 + 1
  -- Perpendicularity condition: product of slopes is -1
  2*a * 2*b = -1 →
  -- P is the intersection point of tangents
  let P : ℝ × ℝ := ((a + b) / 2, tangent_A ((a + b) / 2))
  -- The y-coordinate of P is 3/4
  P.2 = 3/4 := by sorry


end NUMINAMATH_CALUDE_tangent_intersection_y_coordinate_l562_56299


namespace NUMINAMATH_CALUDE_triangle_exists_from_altitudes_l562_56233

/-- Given three positive real numbers, prove that a triangle with these numbers as its altitudes exists. -/
theorem triangle_exists_from_altitudes (h_a h_b h_c : ℝ) 
  (h_pos_a : h_a > 0) (h_pos_b : h_b > 0) (h_pos_c : h_c > 0) : 
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    h_a = (2 * (a * b * c) / (a * b + b * c + c * a)) / a ∧
    h_b = (2 * (a * b * c) / (a * b + b * c + c * a)) / b ∧
    h_c = (2 * (a * b * c) / (a * b + b * c + c * a)) / c :=
sorry

end NUMINAMATH_CALUDE_triangle_exists_from_altitudes_l562_56233


namespace NUMINAMATH_CALUDE_always_greater_than_m_l562_56208

theorem always_greater_than_m (m : ℚ) : m + 2 > m := by
  sorry

end NUMINAMATH_CALUDE_always_greater_than_m_l562_56208


namespace NUMINAMATH_CALUDE_probability_yellow_ball_l562_56262

def num_red_balls : ℕ := 4
def num_yellow_balls : ℕ := 7

def total_balls : ℕ := num_red_balls + num_yellow_balls

theorem probability_yellow_ball :
  (num_yellow_balls : ℚ) / (total_balls : ℚ) = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_ball_l562_56262


namespace NUMINAMATH_CALUDE_count_ones_in_500_pages_l562_56217

/-- Count the occurrences of digit 1 in a number -/
def countOnesInNumber (n : ℕ) : ℕ := sorry

/-- Count the occurrences of digit 1 in page numbers from 1 to n -/
def countOnesInPages (n : ℕ) : ℕ :=
  (List.range n).map countOnesInNumber |>.sum

theorem count_ones_in_500_pages :
  countOnesInPages 500 = 200 := by sorry

end NUMINAMATH_CALUDE_count_ones_in_500_pages_l562_56217
