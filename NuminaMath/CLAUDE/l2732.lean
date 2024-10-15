import Mathlib

namespace NUMINAMATH_CALUDE_custom_op_difference_l2732_273234

-- Define the custom operation
def customOp (x y : ℝ) : ℝ := x * y - 3 * x

-- State the theorem
theorem custom_op_difference : (customOp 7 4) - (customOp 4 7) = -9 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_difference_l2732_273234


namespace NUMINAMATH_CALUDE_smallest_odd_between_2_and_7_l2732_273270

theorem smallest_odd_between_2_and_7 : 
  ∀ n : ℕ, (2 < n ∧ n < 7 ∧ Odd n) → 3 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_between_2_and_7_l2732_273270


namespace NUMINAMATH_CALUDE_patrol_officer_results_l2732_273224

/-- Represents the travel record of the patrol officer -/
def travel_record : List Int := [10, -8, 6, -13, 7, -12, 3, -3]

/-- Position of the gas station relative to the guard post -/
def gas_station_position : Int := 6

/-- Fuel consumption rate of the motorcycle in liters per kilometer -/
def fuel_consumption_rate : ℚ := 0.05

/-- Calculates the final position of the patrol officer relative to the guard post -/
def final_position (record : List Int) : Int :=
  record.sum

/-- Counts the number of times the patrol officer passes the gas station -/
def gas_station_passes (record : List Int) (gas_station_pos : Int) : Nat :=
  sorry

/-- Calculates the total distance traveled by the patrol officer -/
def total_distance (record : List Int) : Int :=
  record.map (Int.natAbs) |>.sum

/-- Calculates the total fuel consumed during the patrol -/
def total_fuel_consumed (distance : Int) (rate : ℚ) : ℚ :=
  rate * distance.toNat

theorem patrol_officer_results :
  (final_position travel_record = -10) ∧
  (gas_station_passes travel_record gas_station_position = 4) ∧
  (total_fuel_consumed (total_distance travel_record) fuel_consumption_rate = 3.1) :=
sorry

end NUMINAMATH_CALUDE_patrol_officer_results_l2732_273224


namespace NUMINAMATH_CALUDE_division_problem_l2732_273214

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 12401 →
  divisor = 163 →
  remainder = 13 →
  dividend = divisor * quotient + remainder →
  quotient = 76 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2732_273214


namespace NUMINAMATH_CALUDE_smith_initial_markers_l2732_273272

/-- The number of new boxes of markers Mr. Smith buys -/
def new_boxes : ℕ := 6

/-- The number of markers in each new box -/
def markers_per_box : ℕ := 9

/-- The total number of markers Mr. Smith has after buying new boxes -/
def total_markers : ℕ := 86

/-- The number of markers Mr. Smith had initially -/
def initial_markers : ℕ := total_markers - (new_boxes * markers_per_box)

theorem smith_initial_markers :
  initial_markers = 32 := by sorry

end NUMINAMATH_CALUDE_smith_initial_markers_l2732_273272


namespace NUMINAMATH_CALUDE_total_individual_packs_l2732_273264

def cookies_packs : ℕ := 3
def cookies_per_pack : ℕ := 4
def noodles_packs : ℕ := 4
def noodles_per_pack : ℕ := 8
def juice_packs : ℕ := 5
def juice_per_pack : ℕ := 6
def snacks_packs : ℕ := 2
def snacks_per_pack : ℕ := 10

theorem total_individual_packs :
  cookies_packs * cookies_per_pack +
  noodles_packs * noodles_per_pack +
  juice_packs * juice_per_pack +
  snacks_packs * snacks_per_pack = 94 := by
  sorry

end NUMINAMATH_CALUDE_total_individual_packs_l2732_273264


namespace NUMINAMATH_CALUDE_angle_inequality_equivalence_l2732_273285

theorem angle_inequality_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ < 0) ↔
  (Real.pi / 2 < θ ∧ θ < 3 * Real.pi / 2) := by sorry

end NUMINAMATH_CALUDE_angle_inequality_equivalence_l2732_273285


namespace NUMINAMATH_CALUDE_sqrt_225_equals_15_l2732_273296

theorem sqrt_225_equals_15 : Real.sqrt 225 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_225_equals_15_l2732_273296


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2732_273217

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let e := c / a
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → 
    ((x + c)^2 + y^2 = (a + e * x)^2) ∧
    (0 - b)^2 / ((-c) - 0)^2 + (b - 0)^2 / (0 - a)^2 = 1) →
  e = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2732_273217


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l2732_273218

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_problem : Nat.gcd (factorial 7) ((factorial 12) / (factorial 5)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l2732_273218


namespace NUMINAMATH_CALUDE_graph_symmetry_l2732_273284

-- Define a general real-valued function
variable (f : ℝ → ℝ)

-- Define the symmetry property about the y-axis
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Theorem statement
theorem graph_symmetry (f : ℝ → ℝ) : 
  symmetric_about_y_axis f ↔ 
  ∀ x y : ℝ, (x, y) ∈ (Set.range (λ x => (x, f x))) ↔ 
              (-x, y) ∈ (Set.range (λ x => (x, f x))) :=
sorry

end NUMINAMATH_CALUDE_graph_symmetry_l2732_273284


namespace NUMINAMATH_CALUDE_square_sum_value_l2732_273291

theorem square_sum_value (a b : ℝ) : 
  (a^2 + b^2 + 2) * (a^2 + b^2 - 2) = 45 → a^2 + b^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l2732_273291


namespace NUMINAMATH_CALUDE_equation_solutions_l2732_273212

theorem equation_solutions :
  ∀ x y : ℤ, y ≥ 0 → (24 * y + 1 = (4 * y^2 - x^2)^2) →
    ((x = 1 ∨ x = -1) ∧ y = 0) ∨
    ((x = 3 ∨ x = -3) ∧ y = 1) ∨
    ((x = 3 ∨ x = -3) ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2732_273212


namespace NUMINAMATH_CALUDE_fraction_simplification_l2732_273202

theorem fraction_simplification :
  (3/7 + 5/8 + 2/9) / (5/12 + 1/4) = 643/336 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2732_273202


namespace NUMINAMATH_CALUDE_younger_brother_height_l2732_273268

theorem younger_brother_height (h1 h2 : ℝ) (h1_positive : 0 < h1) (h2_positive : 0 < h2) 
  (height_difference : h2 - h1 = 12) (height_sum : h1 + h2 = 308) (h1_smaller : h1 < h2) : h1 = 148 :=
by sorry

end NUMINAMATH_CALUDE_younger_brother_height_l2732_273268


namespace NUMINAMATH_CALUDE_power_of_eight_mod_hundred_l2732_273213

theorem power_of_eight_mod_hundred : 8^2050 % 100 = 24 := by sorry

end NUMINAMATH_CALUDE_power_of_eight_mod_hundred_l2732_273213


namespace NUMINAMATH_CALUDE_certain_number_divided_by_ten_l2732_273251

theorem certain_number_divided_by_ten (x : ℝ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_divided_by_ten_l2732_273251


namespace NUMINAMATH_CALUDE_function_six_monotonic_intervals_l2732_273238

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * |x^3| - (a/2) * x^2 + (3-a) * |x| + b

theorem function_six_monotonic_intervals (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (-x)) →
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (∀ x : ℝ, x > 0 → (x^2 - a*x + (3-a) = 0 ↔ (x = x₁ ∨ x = x₂)))) →
  2 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_function_six_monotonic_intervals_l2732_273238


namespace NUMINAMATH_CALUDE_max_xy_value_l2732_273200

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2*x + 3*y = 6) :
  ∃ (max_val : ℝ), max_val = 3/2 ∧ ∀ (z : ℝ), x*y ≤ z → z ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l2732_273200


namespace NUMINAMATH_CALUDE_marble_problem_l2732_273256

theorem marble_problem (x : ℕ) : 
  (((x / 2) * (1 / 3)) * (85 / 100) : ℚ) = 432 → x = 3052 :=
by sorry

end NUMINAMATH_CALUDE_marble_problem_l2732_273256


namespace NUMINAMATH_CALUDE_mary_garbage_bill_calculation_l2732_273293

/-- Calculates Mary's garbage bill given the specified conditions -/
def maryGarbageBill (weeksInMonth : ℕ) (trashBinCost recyclingBinCost greenWasteBinCost : ℚ)
  (trashBinCount recyclingBinCount greenWasteBinCount : ℕ) (serviceFee : ℚ)
  (discountRate : ℚ) (inappropriateItemsFine lateFee : ℚ) : ℚ :=
  let weeklyBinCost := trashBinCost * trashBinCount + recyclingBinCost * recyclingBinCount +
                       greenWasteBinCost * greenWasteBinCount
  let monthlyBinCost := weeklyBinCost * weeksInMonth
  let totalBeforeDiscount := monthlyBinCost + serviceFee
  let discountAmount := totalBeforeDiscount * discountRate
  let totalAfterDiscount := totalBeforeDiscount - discountAmount
  totalAfterDiscount + inappropriateItemsFine + lateFee

/-- Theorem stating that Mary's garbage bill is $134.14 under the given conditions -/
theorem mary_garbage_bill_calculation :
  maryGarbageBill 4 10 5 3 2 1 1 15 (18/100) 20 10 = 134.14 := by
  sorry

end NUMINAMATH_CALUDE_mary_garbage_bill_calculation_l2732_273293


namespace NUMINAMATH_CALUDE_fraction_simplification_l2732_273249

theorem fraction_simplification (x : ℝ) : 
  (x^2 + 2*x + 3) / 4 + (3*x - 5) / 6 = (3*x^2 + 12*x - 1) / 12 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2732_273249


namespace NUMINAMATH_CALUDE_remainder_yards_value_l2732_273235

/-- The number of half-marathons Jacob has run -/
def num_half_marathons : ℕ := 15

/-- The length of a half-marathon in miles -/
def half_marathon_miles : ℕ := 13

/-- The additional length of a half-marathon in yards -/
def half_marathon_extra_yards : ℕ := 193

/-- The number of yards in a mile -/
def yards_per_mile : ℕ := 1760

/-- The total distance Jacob has run in yards -/
def total_distance_yards : ℕ :=
  num_half_marathons * (half_marathon_miles * yards_per_mile + half_marathon_extra_yards)

/-- The remainder y in yards when the total distance is expressed as m miles and y yards -/
def remainder_yards : ℕ := total_distance_yards % yards_per_mile

theorem remainder_yards_value : remainder_yards = 1135 := by
  sorry

end NUMINAMATH_CALUDE_remainder_yards_value_l2732_273235


namespace NUMINAMATH_CALUDE_e_value_l2732_273288

-- Define variables
variable (p j t b a e : ℝ)

-- Define conditions
def condition1 : Prop := j = 0.75 * p
def condition2 : Prop := j = 0.8 * t
def condition3 : Prop := t = p * (1 - e / 100)
def condition4 : Prop := b = 1.4 * j
def condition5 : Prop := a = 0.85 * b
def condition6 : Prop := e = 2 * ((p - a) / p) * 100

-- Theorem statement
theorem e_value (h1 : condition1 p j) (h2 : condition2 j t) (h3 : condition3 p t e)
                (h4 : condition4 j b) (h5 : condition5 b a) (h6 : condition6 p a e) :
  e = 21.5 := by
  sorry

end NUMINAMATH_CALUDE_e_value_l2732_273288


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l2732_273210

theorem smallest_dual_base_representation : ∃ (c d : ℕ), 
  c > 3 ∧ d > 3 ∧ 
  3 * c + 4 = 19 ∧ 
  4 * d + 3 = 19 ∧
  (∀ (x c' d' : ℕ), c' > 3 → d' > 3 → 3 * c' + 4 = x → 4 * d' + 3 = x → x ≥ 19) := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l2732_273210


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l2732_273223

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 13) : 
  x^3 + y^3 = 35 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l2732_273223


namespace NUMINAMATH_CALUDE_inequality_proof_l2732_273253

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b > 0) :
  a / b^2 + b / a^2 > 1 / a + 1 / b := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2732_273253


namespace NUMINAMATH_CALUDE_snowfall_difference_l2732_273282

/-- Snowfall difference calculation -/
theorem snowfall_difference (bald_mountain : ℝ) (billy_mountain : ℝ) (mount_pilot : ℝ) :
  bald_mountain = 1.5 →
  billy_mountain = 3.5 →
  mount_pilot = 1.26 →
  (billy_mountain + mount_pilot - bald_mountain) * 100 = 326 := by
  sorry

end NUMINAMATH_CALUDE_snowfall_difference_l2732_273282


namespace NUMINAMATH_CALUDE_bread_distribution_l2732_273281

theorem bread_distribution (a d : ℚ) (h1 : d > 0) 
  (h2 : (a - 2*d) + (a - d) + a + (a + d) + (a + 2*d) = 100)
  (h3 : (1/7) * (a + (a + d) + (a + 2*d)) = (a - 2*d) + (a - d)) :
  a - 2*d = 5/3 := by
sorry

end NUMINAMATH_CALUDE_bread_distribution_l2732_273281


namespace NUMINAMATH_CALUDE_optimal_bicycle_point_l2732_273275

/-- Represents the problem of finding the optimal point to leave a bicycle --/
theorem optimal_bicycle_point 
  (total_distance : ℝ) 
  (cycling_speed walking_speed : ℝ) 
  (h1 : total_distance = 30) 
  (h2 : cycling_speed = 20) 
  (h3 : walking_speed = 5) :
  ∃ (x : ℝ), 
    x = 5 ∧ 
    (∀ (y : ℝ), 0 ≤ y ∧ y ≤ total_distance → 
      max ((total_distance / 2 - y) / cycling_speed + y / walking_speed)
          (y / walking_speed + (total_distance / 2 - y) / cycling_speed)
      ≥ 
      max ((total_distance / 2 - x) / cycling_speed + x / walking_speed)
          (x / walking_speed + (total_distance / 2 - x) / cycling_speed)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_bicycle_point_l2732_273275


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_lambda_l2732_273262

/-- Given two 2D vectors a and b, if a + 3b is parallel to 2a - b, then the second component of b is -8/3 -/
theorem parallel_vectors_imply_lambda (a b : ℝ × ℝ) (h : a = (-3, 2) ∧ b.1 = 4) :
  (∃ (k : ℝ), k ≠ 0 ∧ k • (a + 3 • b) = 2 • a - b) → b.2 = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_lambda_l2732_273262


namespace NUMINAMATH_CALUDE_log_equation_l2732_273230

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation : (log10 5)^2 + log10 2 * log10 50 = 1 := by sorry

end NUMINAMATH_CALUDE_log_equation_l2732_273230


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_is_all_reals_l2732_273246

theorem quadratic_inequality_solution_is_all_reals :
  let f : ℝ → ℝ := λ x => -3 * x^2 + 5 * x + 6
  ∀ x : ℝ, f x < 0 ∨ f x > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_is_all_reals_l2732_273246


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l2732_273298

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l2732_273298


namespace NUMINAMATH_CALUDE_train_length_calculation_l2732_273261

/-- The length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) 
  (h1 : train_speed = 27) 
  (h2 : man_speed = 6) 
  (h3 : passing_time = 11.999040076793857) : 
  ∃ (length : ℝ), abs (length - 110) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_train_length_calculation_l2732_273261


namespace NUMINAMATH_CALUDE_angle_expression_equality_l2732_273292

theorem angle_expression_equality (θ : Real) 
  (h1 : ∃ (x y : Real), x < 0 ∧ y < 0 ∧ Real.cos θ = x ∧ Real.sin θ = y) 
  (h2 : Real.tan θ ^ 2 = -2 * Real.sqrt 2) : 
  Real.sin θ ^ 2 - Real.sin (3 * Real.pi + θ) * Real.cos (Real.pi + θ) - Real.sqrt 2 * Real.cos θ ^ 2 = (2 - 2 * Real.sqrt 2) / 3 := by
  sorry


end NUMINAMATH_CALUDE_angle_expression_equality_l2732_273292


namespace NUMINAMATH_CALUDE_floor_sum_example_l2732_273239

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by sorry

end NUMINAMATH_CALUDE_floor_sum_example_l2732_273239


namespace NUMINAMATH_CALUDE_expand_product_l2732_273227

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2732_273227


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2732_273294

/-- The polynomial function we're considering -/
def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 5*x - 10

/-- The roots of the polynomial -/
def roots : Set ℝ := {-2, Real.sqrt 5, -Real.sqrt 5}

/-- Theorem stating that the given set contains all roots of the polynomial -/
theorem roots_of_polynomial :
  ∀ x : ℝ, f x = 0 ↔ x ∈ roots := by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2732_273294


namespace NUMINAMATH_CALUDE_area_of_similar_rectangle_l2732_273273

/-- Given a rectangle R1 with one side of 3 inches and an area of 18 square inches,
    and a similar rectangle R2 with a diagonal of 18 inches,
    prove that the area of R2 is 14.4 square inches. -/
theorem area_of_similar_rectangle (r1_side : ℝ) (r1_area : ℝ) (r2_diagonal : ℝ) :
  r1_side = 3 →
  r1_area = 18 →
  r2_diagonal = 18 →
  ∃ (r2_side1 r2_side2 : ℝ),
    r2_side1 * r2_side2 = 14.4 ∧
    r2_side1^2 + r2_side2^2 = r2_diagonal^2 ∧
    r2_side2 / r2_side1 = r1_area / r1_side^2 :=
by sorry

end NUMINAMATH_CALUDE_area_of_similar_rectangle_l2732_273273


namespace NUMINAMATH_CALUDE_distribution_theorem_l2732_273252

/-- Represents the number of communities --/
def n : ℕ := 5

/-- Represents the number of fitness equipment --/
def k : ℕ := 7

/-- Represents the number of communities that must receive at least 2 items --/
def m : ℕ := 2

/-- Represents the minimum number of items each of the m communities must receive --/
def min_items : ℕ := 2

/-- The number of ways to distribute k identical items among n recipients,
    where m specific recipients must receive at least min_items each --/
def distribution_schemes (n k m min_items : ℕ) : ℕ := sorry

theorem distribution_theorem : distribution_schemes n k m min_items = 35 := by sorry

end NUMINAMATH_CALUDE_distribution_theorem_l2732_273252


namespace NUMINAMATH_CALUDE_cycle_sale_result_l2732_273240

/-- Calculates the final selling price and overall profit percentage for a cycle sale --/
def cycle_sale_analysis (initial_cost upgrade_cost : ℚ) (profit_margin sales_tax : ℚ) :
  ℚ × ℚ :=
  let total_cost := initial_cost + upgrade_cost
  let selling_price_before_tax := total_cost * (1 + profit_margin)
  let final_selling_price := selling_price_before_tax * (1 + sales_tax)
  let overall_profit := final_selling_price - total_cost
  let overall_profit_percentage := (overall_profit / total_cost) * 100
  (final_selling_price, overall_profit_percentage)

/-- Theorem stating the correct final selling price and overall profit percentage --/
theorem cycle_sale_result :
  let (final_price, profit_percentage) := cycle_sale_analysis 1400 600 (10/100) (5/100)
  final_price = 2310 ∧ profit_percentage = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_cycle_sale_result_l2732_273240


namespace NUMINAMATH_CALUDE_ellipse_from_hyperbola_l2732_273257

/-- The hyperbola from which we derive the ellipse parameters -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = -1

/-- The vertex of the hyperbola becomes the focus of the ellipse -/
def hyperbola_vertex_to_ellipse_focus (x y : ℝ) : Prop :=
  hyperbola x y → (x = 0 ∧ (y = 2 ∨ y = -2))

/-- The focus of the hyperbola becomes the vertex of the ellipse -/
def hyperbola_focus_to_ellipse_vertex (x y : ℝ) : Prop :=
  hyperbola x y → (x = 0 ∧ (y = 4 ∨ y = -4))

/-- The resulting ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 16 = 1

/-- Theorem stating that the ellipse with the given properties has the specified equation -/
theorem ellipse_from_hyperbola :
  (∀ x y, hyperbola_vertex_to_ellipse_focus x y) →
  (∀ x y, hyperbola_focus_to_ellipse_vertex x y) →
  (∀ x y, ellipse x y) :=
sorry

end NUMINAMATH_CALUDE_ellipse_from_hyperbola_l2732_273257


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2732_273216

theorem polynomial_factorization (x : ℤ) :
  9 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 5 * x^2 =
  (3 * x^2 + 59 * x + 231) * (x + 7) * (3 * x + 11) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2732_273216


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2732_273259

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The problem statement -/
theorem perpendicular_line_through_point :
  ∃ (l : Line),
    perpendicular l (Line.mk 1 (-2) (-1)) ∧
    point_on_line 1 1 l ∧
    l = Line.mk 2 1 (-3) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2732_273259


namespace NUMINAMATH_CALUDE_madeline_work_hours_l2732_273289

def monthly_expenses : ℕ := 1200 + 400 + 200 + 60
def emergency_savings : ℕ := 200
def daytime_hourly_rate : ℕ := 15
def bakery_hourly_rate : ℕ := 12
def bakery_weekly_hours : ℕ := 5
def tax_rate : ℚ := 15 / 100

theorem madeline_work_hours :
  ∃ (h : ℕ), h ≥ 146 ∧
  (h * daytime_hourly_rate + 4 * bakery_weekly_hours * bakery_hourly_rate) * (1 - tax_rate) ≥ 
  (monthly_expenses + emergency_savings : ℚ) ∧
  ∀ (k : ℕ), k < h →
  (k * daytime_hourly_rate + 4 * bakery_weekly_hours * bakery_hourly_rate) * (1 - tax_rate) <
  (monthly_expenses + emergency_savings : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_madeline_work_hours_l2732_273289


namespace NUMINAMATH_CALUDE_correct_num_ways_to_choose_l2732_273250

/-- The number of humanities courses -/
def num_humanities : ℕ := 4

/-- The number of natural science courses -/
def num_sciences : ℕ := 3

/-- The total number of courses to be chosen -/
def courses_to_choose : ℕ := 3

/-- The number of conflicting course pairs (A₁ and B₁) -/
def num_conflicts : ℕ := 1

/-- The function that calculates the number of ways to choose courses -/
def num_ways_to_choose : ℕ := sorry

theorem correct_num_ways_to_choose :
  num_ways_to_choose = 25 := by sorry

end NUMINAMATH_CALUDE_correct_num_ways_to_choose_l2732_273250


namespace NUMINAMATH_CALUDE_asymptotes_of_hyperbola_l2732_273276

/-- Hyperbola C with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ 0 < b

/-- Point on the right branch of hyperbola C -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1
  h_right_branch : 0 < x

/-- Equilateral triangle with vertices on hyperbola -/
structure EquilateralTriangleOnHyperbola (h : Hyperbola) where
  A : PointOnHyperbola h
  B : PointOnHyperbola h
  c : ℝ
  h_equilateral : c^2 = A.x^2 + A.y^2 ∧ c^2 = B.x^2 + B.y^2
  h_side_length : c^2 = h.a^2 + h.b^2

/-- Theorem: Asymptotes of hyperbola C are y = ±x -/
theorem asymptotes_of_hyperbola (h : Hyperbola) 
  (t : EquilateralTriangleOnHyperbola h) :
  ∃ (k : ℝ), k = 1 ∧ 
  (∀ (x y : ℝ), (y = k*x ∨ y = -k*x) ↔ 
    (∀ ε > 0, ∃ δ > 0, ∀ x' y', 
      x'^2/h.a^2 - y'^2/h.b^2 = 1 → 
      x' > δ → |y'/x' - k| < ε ∨ |y'/x' + k| < ε)) :=
sorry

end NUMINAMATH_CALUDE_asymptotes_of_hyperbola_l2732_273276


namespace NUMINAMATH_CALUDE_egg_packing_problem_l2732_273258

/-- The number of baskets containing eggs -/
def num_baskets : ℕ := 21

/-- The number of eggs in each basket -/
def eggs_per_basket : ℕ := 48

/-- The number of eggs each box can hold -/
def eggs_per_box : ℕ := 28

/-- The number of boxes needed to pack all the eggs -/
def boxes_needed : ℕ := (num_baskets * eggs_per_basket) / eggs_per_box

theorem egg_packing_problem : boxes_needed = 36 := by
  sorry

end NUMINAMATH_CALUDE_egg_packing_problem_l2732_273258


namespace NUMINAMATH_CALUDE_auto_dealer_sales_l2732_273221

theorem auto_dealer_sales (trucks : ℕ) (cars : ℕ) : 
  trucks = 21 →
  cars = trucks + 27 →
  cars + trucks = 69 := by
sorry

end NUMINAMATH_CALUDE_auto_dealer_sales_l2732_273221


namespace NUMINAMATH_CALUDE_product_zero_l2732_273220

theorem product_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l2732_273220


namespace NUMINAMATH_CALUDE_sum_of_parts_of_complex_number_l2732_273260

theorem sum_of_parts_of_complex_number : ∃ (z : ℂ), 
  z = (Complex.I * 2 - 3) * (Complex.I - 2) / Complex.I ∧ 
  z.re + z.im = -11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_of_complex_number_l2732_273260


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_l2732_273280

theorem absolute_value_nonnegative (x : ℝ) : ¬(|x| < 0) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_l2732_273280


namespace NUMINAMATH_CALUDE_largest_product_of_three_exists_product_72_largest_product_is_72_l2732_273225

def S : Finset Int := {-4, -3, -1, 5, 6}

theorem largest_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (a * b * c : Int) ≤ 72 :=
sorry

theorem exists_product_72 : 
  ∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 72 :=
sorry

theorem largest_product_is_72 : 
  (∀ (a b c : Int), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → (a * b * c : Int) ≤ 72) ∧
  (∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 72) :=
sorry

end NUMINAMATH_CALUDE_largest_product_of_three_exists_product_72_largest_product_is_72_l2732_273225


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2732_273241

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2732_273241


namespace NUMINAMATH_CALUDE_graph_horizontal_shift_l2732_273254

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define a point (x, y) on the original graph
variable (x y : ℝ)

-- Define the horizontal shift
def h : ℝ := 2

-- Theorem stating that y = f(x + 2) is equivalent to shifting the graph of y = f(x) 2 units left
theorem graph_horizontal_shift :
  y = f (x + h) ↔ y = f ((x + h) - h) :=
sorry

end NUMINAMATH_CALUDE_graph_horizontal_shift_l2732_273254


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l2732_273243

/-- Proves that under given conditions, the cost increase percentage is 25% -/
theorem cost_increase_percentage (C : ℝ) (P : ℝ) : 
  C > 0 → -- Ensure cost is positive
  let S := 4.2 * C -- Original selling price
  let new_profit := 0.7023809523809523 * S -- New profit after cost increase
  3.2 * C - (P / 100) * C = new_profit → -- Equation relating new profit to cost increase
  P = 25 := by
  sorry

end NUMINAMATH_CALUDE_cost_increase_percentage_l2732_273243


namespace NUMINAMATH_CALUDE_saree_price_calculation_l2732_273278

theorem saree_price_calculation (P : ℝ) : 
  (P * (1 - 0.15) * (1 - 0.05) = 323) → P = 400 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l2732_273278


namespace NUMINAMATH_CALUDE_composition_ratio_l2732_273242

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio : f (g (f 2)) / g (f (g 2)) = 115 / 73 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l2732_273242


namespace NUMINAMATH_CALUDE_alexis_has_60_mangoes_l2732_273245

/-- Represents the number of mangoes each person has -/
structure MangoDistribution where
  alexis : ℕ
  dilan : ℕ
  ashley : ℕ

/-- Defines the conditions of the mango distribution problem -/
def validDistribution (d : MangoDistribution) : Prop :=
  (d.alexis = 4 * (d.dilan + d.ashley)) ∧
  (d.alexis + d.dilan + d.ashley = 75)

/-- Theorem stating that Alexis has 60 mangoes in a valid distribution -/
theorem alexis_has_60_mangoes (d : MangoDistribution) 
  (h : validDistribution d) : d.alexis = 60 := by
  sorry

end NUMINAMATH_CALUDE_alexis_has_60_mangoes_l2732_273245


namespace NUMINAMATH_CALUDE_beidou_timing_accuracy_l2732_273265

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem beidou_timing_accuracy : 
  toScientificNotation 0.0000000099 = ScientificNotation.mk 9.9 (-9) sorry := by
  sorry

end NUMINAMATH_CALUDE_beidou_timing_accuracy_l2732_273265


namespace NUMINAMATH_CALUDE_max_points_tournament_l2732_273236

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (games_per_pair : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculate the total number of games in the tournament -/
def total_games (t : Tournament) : ℕ :=
  t.num_teams.choose 2 * t.games_per_pair

/-- Calculate the maximum points achievable by the top three teams -/
def max_points_top_three (t : Tournament) : ℕ :=
  let games_against_lower := (t.num_teams - 3) * t.games_per_pair
  let points_from_lower := games_against_lower * t.points_for_win
  let games_among_top := 2 * t.games_per_pair
  let points_among_top := games_among_top * t.points_for_win / 2
  points_from_lower + points_among_top

/-- The main theorem stating the maximum points for top three teams -/
theorem max_points_tournament :
  ∀ t : Tournament,
    t.num_teams = 8 →
    t.games_per_pair = 2 →
    t.points_for_win = 3 →
    t.points_for_draw = 1 →
    t.points_for_loss = 0 →
    max_points_top_three t = 36 := by
  sorry

end NUMINAMATH_CALUDE_max_points_tournament_l2732_273236


namespace NUMINAMATH_CALUDE_tv_show_cost_l2732_273201

/-- Calculates the total cost of a TV show season -/
def season_cost (total_episodes : ℕ) (first_half_cost : ℝ) (second_half_increase : ℝ) : ℝ :=
  let half_episodes := total_episodes / 2
  let first_half_total := first_half_cost * half_episodes
  let second_half_cost := first_half_cost * (1 + second_half_increase)
  let second_half_total := second_half_cost * half_episodes
  first_half_total + second_half_total

/-- Theorem stating the total cost of the TV show season -/
theorem tv_show_cost : 
  season_cost 22 1000 1.2 = 35200 := by
  sorry

#eval season_cost 22 1000 1.2

end NUMINAMATH_CALUDE_tv_show_cost_l2732_273201


namespace NUMINAMATH_CALUDE_chord_length_l2732_273297

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l2732_273297


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2732_273209

theorem no_integer_solutions (n : ℤ) : ¬ ∃ x : ℤ, x^2 - 16*n*x + 7^5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2732_273209


namespace NUMINAMATH_CALUDE_minuend_value_l2732_273247

theorem minuend_value (minuend subtrahend : ℝ) 
  (h : minuend + subtrahend + (minuend - subtrahend) = 25) : 
  minuend = 12.5 := by
sorry

end NUMINAMATH_CALUDE_minuend_value_l2732_273247


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l2732_273222

-- Define the room dimensions and paving rate
def room_length : ℝ := 5.5
def room_width : ℝ := 4
def paving_rate : ℝ := 750

-- Define the function to calculate the cost of paving
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

-- State the theorem
theorem paving_cost_calculation :
  paving_cost room_length room_width paving_rate = 16500 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l2732_273222


namespace NUMINAMATH_CALUDE_smallest_factorial_divisible_by_7875_l2732_273228

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_factorial_divisible_by_7875 :
  ∃ (n : ℕ), (n > 0) ∧ (is_factor 7875 (Nat.factorial n)) ∧
  (∀ (m : ℕ), m > 0 → m < n → ¬(is_factor 7875 (Nat.factorial m))) ∧
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_smallest_factorial_divisible_by_7875_l2732_273228


namespace NUMINAMATH_CALUDE_john_videos_per_day_l2732_273229

/-- Represents the number of videos and their durations for a video creator --/
structure VideoCreator where
  short_videos_per_day : ℕ
  long_videos_per_day : ℕ
  short_video_duration : ℕ
  long_video_duration : ℕ
  days_per_week : ℕ
  total_weekly_minutes : ℕ

/-- Calculates the total number of videos released per day --/
def total_videos_per_day (vc : VideoCreator) : ℕ :=
  vc.short_videos_per_day + vc.long_videos_per_day

/-- Calculates the total minutes of video released per day --/
def total_minutes_per_day (vc : VideoCreator) : ℕ :=
  vc.short_videos_per_day * vc.short_video_duration +
  vc.long_videos_per_day * vc.long_video_duration

/-- Theorem stating that given the conditions, the total number of videos released per day is 3 --/
theorem john_videos_per_day :
  ∀ (vc : VideoCreator),
  vc.short_videos_per_day = 2 →
  vc.long_videos_per_day = 1 →
  vc.short_video_duration = 2 →
  vc.long_video_duration = 6 * vc.short_video_duration →
  vc.days_per_week = 7 →
  vc.total_weekly_minutes = 112 →
  vc.total_weekly_minutes = vc.days_per_week * (total_minutes_per_day vc) →
  total_videos_per_day vc = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_videos_per_day_l2732_273229


namespace NUMINAMATH_CALUDE_units_digit_of_product_units_digit_27_times_46_l2732_273226

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The property that the units digit of a product depends only on the units digits of its factors -/
theorem units_digit_of_product (a b : ℕ) :
  unitsDigit (a * b) = unitsDigit (unitsDigit a * unitsDigit b) := by
  sorry

/-- The main theorem: the units digit of 27 * 46 is equal to the units digit of 7 * 6 -/
theorem units_digit_27_times_46 :
  unitsDigit (27 * 46) = unitsDigit (7 * 6) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_units_digit_27_times_46_l2732_273226


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2732_273269

/-- 
Given a system of linear equations:
  a * x + b * y - b * z = c
  a * y + b * x - b * z = c
  a * z + b * y - b * x = c
This theorem states that the system has a unique solution if and only if 
a ≠ 0, a - b ≠ 0, and a + b ≠ 0.
-/
theorem unique_solution_condition (a b c : ℝ) :
  (∃! x y z : ℝ, (a * x + b * y - b * z = c) ∧ 
                 (a * y + b * x - b * z = c) ∧ 
                 (a * z + b * y - b * x = c)) ↔ 
  (a ≠ 0 ∧ a - b ≠ 0 ∧ a + b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2732_273269


namespace NUMINAMATH_CALUDE_reader_group_total_l2732_273287

/-- Represents the number of readers in a group reading different types of books. -/
structure ReaderGroup where
  sci_fi : ℕ     -- Number of readers who read science fiction
  literary : ℕ   -- Number of readers who read literary works
  both : ℕ       -- Number of readers who read both

/-- Calculates the total number of readers in the group. -/
def total_readers (g : ReaderGroup) : ℕ :=
  g.sci_fi + g.literary - g.both

/-- Theorem stating that for the given reader numbers, the total is 650. -/
theorem reader_group_total :
  ∃ (g : ReaderGroup), g.sci_fi = 250 ∧ g.literary = 550 ∧ g.both = 150 ∧ total_readers g = 650 :=
by
  sorry

#check reader_group_total

end NUMINAMATH_CALUDE_reader_group_total_l2732_273287


namespace NUMINAMATH_CALUDE_sum_quadratic_residues_divisible_l2732_273244

theorem sum_quadratic_residues_divisible (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ s : ℕ, (s > 0) ∧ (s < p) ∧ (∀ x : ℕ, x < p → (∃ y : ℕ, y < p ∧ y^2 ≡ x [ZMOD p]) → s ≡ s + x [ZMOD p]) :=
sorry

end NUMINAMATH_CALUDE_sum_quadratic_residues_divisible_l2732_273244


namespace NUMINAMATH_CALUDE_three_points_distance_is_four_l2732_273286

/-- The quadratic function f(x) = x^2 - 2x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

/-- A point (x, y) is on the graph of f if y = f(x) -/
def on_graph (x y : ℝ) : Prop := y = f x

/-- The distance of a point (x, y) from the x-axis is the absolute value of y -/
def distance_from_x_axis (y : ℝ) : ℝ := |y|

/-- There exist exactly three points on the graph of f with distance m from the x-axis -/
def three_points_with_distance (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    on_graph x₁ y₁ ∧ on_graph x₂ y₂ ∧ on_graph x₃ y₃ ∧
    distance_from_x_axis y₁ = m ∧
    distance_from_x_axis y₂ = m ∧
    distance_from_x_axis y₃ = m ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
    ∀ x y : ℝ, on_graph x y → distance_from_x_axis y = m → (x = x₁ ∨ x = x₂ ∨ x = x₃)

theorem three_points_distance_is_four :
  ∀ m : ℝ, three_points_with_distance m → m = 4 :=
by sorry

end NUMINAMATH_CALUDE_three_points_distance_is_four_l2732_273286


namespace NUMINAMATH_CALUDE_absolute_value_minus_sqrt_l2732_273274

theorem absolute_value_minus_sqrt (a : ℝ) (h : a < -1) : |1 + a| - Real.sqrt (a^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_minus_sqrt_l2732_273274


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2732_273231

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 81

-- Define the factored form
def f (x : ℝ) : ℝ := (x-3)*(x+3)*(x^2+9)

-- Theorem stating the equality of the polynomial and its factored form
theorem polynomial_factorization :
  ∀ x : ℝ, p x = f x :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2732_273231


namespace NUMINAMATH_CALUDE_problem_statement_l2732_273271

theorem problem_statement (x y : ℝ) 
  (hx : x = 1 / (Real.sqrt 2 + 1)) 
  (hy : y = 1 / (Real.sqrt 2 - 1)) : 
  x^2 - 3*x*y + y^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2732_273271


namespace NUMINAMATH_CALUDE_import_tax_calculation_l2732_273205

theorem import_tax_calculation (tax_rate : ℝ) (tax_threshold : ℝ) (tax_paid : ℝ) (total_value : ℝ) : 
  tax_rate = 0.07 →
  tax_threshold = 1000 →
  tax_paid = 87.50 →
  tax_rate * (total_value - tax_threshold) = tax_paid →
  total_value = 2250 := by
sorry

end NUMINAMATH_CALUDE_import_tax_calculation_l2732_273205


namespace NUMINAMATH_CALUDE_no_x_squared_term_l2732_273204

theorem no_x_squared_term (a : ℝ) : 
  (∀ x, (x + 1) * (x^2 - 5*a*x + a) = x^3 + (-4*a)*x + a) → a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l2732_273204


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2732_273277

theorem arithmetic_mean_of_fractions :
  (1 / 3 : ℚ) * (3 / 7 + 5 / 9 + 7 / 11) = 1123 / 2079 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2732_273277


namespace NUMINAMATH_CALUDE_paper_stack_height_l2732_273266

/-- Given a stack of paper where 800 sheets are 4 cm thick, 
    prove that a 6 cm high stack would contain 1200 sheets. -/
theorem paper_stack_height (sheets : ℕ) (height : ℝ) : 
  (800 : ℝ) / 4 = sheets / height → sheets = 1200 ∧ height = 6 :=
by sorry

end NUMINAMATH_CALUDE_paper_stack_height_l2732_273266


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l2732_273279

theorem gcd_lcm_sum : Nat.gcd 24 54 + Nat.lcm 48 18 = 150 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l2732_273279


namespace NUMINAMATH_CALUDE_runner_problem_l2732_273203

theorem runner_problem (v : ℝ) (h1 : v > 0) :
  (40 / v = 20 / v + 4) →
  (40 / (v / 2) = 8) :=
by sorry

end NUMINAMATH_CALUDE_runner_problem_l2732_273203


namespace NUMINAMATH_CALUDE_sally_grew_113_turnips_l2732_273211

/-- The number of turnips Sally grew -/
def sallys_turnips : ℕ := 113

/-- The number of pumpkins Sally grew -/
def sallys_pumpkins : ℕ := 118

/-- The number of turnips Mary grew -/
def marys_turnips : ℕ := 129

/-- The total number of turnips grown by Sally and Mary -/
def total_turnips : ℕ := 242

/-- Theorem stating that Sally grew 113 turnips -/
theorem sally_grew_113_turnips :
  sallys_turnips = total_turnips - marys_turnips :=
by sorry

end NUMINAMATH_CALUDE_sally_grew_113_turnips_l2732_273211


namespace NUMINAMATH_CALUDE_factor_implies_p_value_l2732_273255

theorem factor_implies_p_value (m p : ℤ) : 
  (m - 8) ∣ (m^2 - p*m - 24) → p = 5 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_p_value_l2732_273255


namespace NUMINAMATH_CALUDE_sin_alpha_value_l2732_273219

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α - π / 3) = 2 / 3) : 
  Real.sin α = (2 * Real.sqrt 3 - Real.sqrt 5) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l2732_273219


namespace NUMINAMATH_CALUDE_average_weight_increase_l2732_273237

theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 8 * initial_average
  let new_total := initial_total - 50 + 70
  let new_average := new_total / 8
  new_average - initial_average = 2.5 := by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2732_273237


namespace NUMINAMATH_CALUDE_fraction_power_product_l2732_273208

theorem fraction_power_product :
  (1 / 3 : ℚ) ^ 4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l2732_273208


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2732_273290

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2003 + a 2005 + a 2007 + a 2009 + a 2011 + a 2013 = 120) :
  2 * a 2018 - a 2028 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2732_273290


namespace NUMINAMATH_CALUDE_gcd_max_value_l2732_273248

theorem gcd_max_value (m : ℕ+) : 
  Nat.gcd (14 * m.val + 4) (9 * m.val + 2) ≤ 8 ∧ 
  ∃ n : ℕ+, Nat.gcd (14 * n.val + 4) (9 * n.val + 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_max_value_l2732_273248


namespace NUMINAMATH_CALUDE_choir_average_age_l2732_273283

theorem choir_average_age
  (num_females : ℕ)
  (num_males : ℕ)
  (avg_age_females : ℚ)
  (avg_age_males : ℚ)
  (total_people : ℕ)
  (h1 : num_females = 12)
  (h2 : num_males = 15)
  (h3 : avg_age_females = 28)
  (h4 : avg_age_males = 35)
  (h5 : total_people = num_females + num_males) :
  (num_females * avg_age_females + num_males * avg_age_males) / total_people = 31.89 := by
  sorry

end NUMINAMATH_CALUDE_choir_average_age_l2732_273283


namespace NUMINAMATH_CALUDE_exist_three_numbers_equal_sum_l2732_273263

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: Existence of three different natural numbers with equal sum of number and its digits -/
theorem exist_three_numbers_equal_sum :
  ∃ (m n p : ℕ), m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ m + S m = n + S n ∧ n + S n = p + S p :=
sorry

end NUMINAMATH_CALUDE_exist_three_numbers_equal_sum_l2732_273263


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l2732_273215

/-- The last two nonzero digits of n! -/
def lastTwoNonzeroDigits (n : ℕ) : ℕ := sorry

/-- Theorem: The last two nonzero digits of 80! are 08 -/
theorem last_two_nonzero_digits_80_factorial :
  lastTwoNonzeroDigits 80 = 8 := by sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l2732_273215


namespace NUMINAMATH_CALUDE_passes_through_origin_symmetric_about_y_axis_l2732_273206

-- Define the quadratic function
def f (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + m - 2

-- Theorem for passing through the origin
theorem passes_through_origin (m : ℝ) : 
  (f m 0 = 0) ↔ (m = 1 ∨ m = -2) := by sorry

-- Theorem for symmetry about y-axis
theorem symmetric_about_y_axis (m : ℝ) :
  (∀ x, f m x = f m (-x)) ↔ m = 0 := by sorry

end NUMINAMATH_CALUDE_passes_through_origin_symmetric_about_y_axis_l2732_273206


namespace NUMINAMATH_CALUDE_triangle_problem_l2732_273295

/-- Triangle sum for the nth row -/
def triangle_sum (n : ℕ) (a d : ℕ) : ℕ := 2^n * a + (2^n - 2) * d

/-- The problem statement -/
theorem triangle_problem (a d : ℕ) (ha : a > 0) (hd : d > 0) :
  (∃ n : ℕ, triangle_sum n a d = 1988) →
  (∃ n : ℕ, n = 6 ∧ a = 2 ∧ d = 30 ∧ 
    (∀ m : ℕ, triangle_sum m a d = 1988 → m ≤ n)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2732_273295


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l2732_273207

theorem rhombus_diagonal (A : ℝ) (d : ℝ) : 
  d > 0 →  -- shorter diagonal is positive
  3 * d > 0 →  -- longer diagonal is positive
  A = (1/2) * d * (3*d) →  -- area formula
  40 = 4 * (((d/2)^2 + ((3*d)/2)^2)^(1/2)) →  -- perimeter formula
  d = (1/3) * (10 * A)^(1/2) := by sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l2732_273207


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l2732_273232

theorem solution_of_linear_equation :
  ∃ x : ℝ, 2 * x + 6 = 0 ∧ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l2732_273232


namespace NUMINAMATH_CALUDE_hypotenuse_of_right_isosceles_triangle_l2732_273267

-- Define the triangle
def right_isosceles_triangle (leg : ℝ) (hypotenuse : ℝ) : Prop :=
  leg > 0 ∧ hypotenuse > 0 ∧ hypotenuse^2 = 2 * leg^2

-- Theorem statement
theorem hypotenuse_of_right_isosceles_triangle :
  ∀ (leg : ℝ) (hypotenuse : ℝ),
  right_isosceles_triangle leg hypotenuse →
  leg = 8 →
  hypotenuse = 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_of_right_isosceles_triangle_l2732_273267


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2732_273299

/-- Given two arithmetic sequences, prove the ratio of their 4th terms -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) : 
  (∀ n, S n / T n = (7 * n + 2) / (n + 3)) →  -- Given condition
  (∀ n, S n = (a 1 + a n) * n / 2) →  -- Definition of S_n for arithmetic sequence
  (∀ n, T n = (b 1 + b n) * n / 2) →  -- Definition of T_n for arithmetic sequence
  a 4 / b 4 = 51 / 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2732_273299


namespace NUMINAMATH_CALUDE_sum_of_roots_l2732_273233

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x : ℝ, x^2 - 10*a*x - 11*b = 0 ↔ (x = c ∨ x = d)) →
  (∀ x : ℝ, x^2 - 10*c*x - 11*d = 0 ↔ (x = a ∨ x = b)) →
  a + b + c + d = 1210 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2732_273233
