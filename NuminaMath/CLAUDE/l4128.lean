import Mathlib

namespace NUMINAMATH_CALUDE_initial_bacteria_count_l4128_412813

def bacteria_growth (initial_count : ℕ) (time : ℕ) : ℕ :=
  initial_count * 4^(time / 30)

theorem initial_bacteria_count :
  ∃ (initial_count : ℕ),
    bacteria_growth initial_count 360 = 262144 ∧
    initial_count = 1 :=
by sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l4128_412813


namespace NUMINAMATH_CALUDE_point_on_y_axis_l4128_412866

/-- A point P with coordinates (m+2, 2m-4) that lies on the y-axis has coordinates (0, -8). -/
theorem point_on_y_axis (m : ℝ) :
  (m + 2 = 0) → (m + 2, 2 * m - 4) = (0, -8) := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l4128_412866


namespace NUMINAMATH_CALUDE_octal_addition_521_146_l4128_412854

/-- Represents an octal number as a list of digits (0-7) in reverse order --/
def OctalNumber := List Nat

/-- Converts an octal number to its decimal representation --/
def octal_to_decimal (n : OctalNumber) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

/-- Adds two octal numbers and returns the result in octal --/
def add_octal (a b : OctalNumber) : OctalNumber :=
  sorry

theorem octal_addition_521_146 :
  let a : OctalNumber := [1, 2, 5]  -- 521₈ in reverse order
  let b : OctalNumber := [6, 4, 1]  -- 146₈ in reverse order
  let result : OctalNumber := [7, 6, 6]  -- 667₈ in reverse order
  add_octal a b = result :=
by sorry

end NUMINAMATH_CALUDE_octal_addition_521_146_l4128_412854


namespace NUMINAMATH_CALUDE_division_of_decimals_l4128_412805

theorem division_of_decimals : (0.2 : ℚ) / (0.005 : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_division_of_decimals_l4128_412805


namespace NUMINAMATH_CALUDE_zero_points_sum_gt_one_l4128_412816

theorem zero_points_sum_gt_one (x₁ x₂ m : ℝ) 
  (h₁ : x₁ < x₂) 
  (h₂ : Real.log x₁ + 1 / (2 * x₁) = m) 
  (h₃ : Real.log x₂ + 1 / (2 * x₂) = m) : 
  x₁ + x₂ > 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_points_sum_gt_one_l4128_412816


namespace NUMINAMATH_CALUDE_pizza_combinations_l4128_412802

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  (n.choose 1) + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l4128_412802


namespace NUMINAMATH_CALUDE_distance_equality_l4128_412862

theorem distance_equality : ∃ x : ℝ, |x - (-2)| = |x - 4| :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_distance_equality_l4128_412862


namespace NUMINAMATH_CALUDE_value_of_x_l4128_412812

theorem value_of_x : ∀ (x a b c : ℤ),
  x = a + 7 →
  a = b + 12 →
  b = c + 25 →
  c = 95 →
  x = 139 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l4128_412812


namespace NUMINAMATH_CALUDE_tv_watching_time_l4128_412896

/-- The number of episodes of Jeopardy watched -/
def jeopardy_episodes : ℕ := 2

/-- The number of episodes of Wheel of Fortune watched -/
def wheel_episodes : ℕ := 2

/-- The duration of one episode of Jeopardy in minutes -/
def jeopardy_duration : ℕ := 20

/-- The duration of one episode of Wheel of Fortune in minutes -/
def wheel_duration : ℕ := 2 * jeopardy_duration

/-- The total time spent watching TV in minutes -/
def total_time : ℕ := jeopardy_episodes * jeopardy_duration + wheel_episodes * wheel_duration

/-- Conversion factor from minutes to hours -/
def minutes_per_hour : ℕ := 60

/-- Theorem: James watched TV for 2 hours -/
theorem tv_watching_time : total_time / minutes_per_hour = 2 := by
  sorry

end NUMINAMATH_CALUDE_tv_watching_time_l4128_412896


namespace NUMINAMATH_CALUDE_calculation_proof_l4128_412884

theorem calculation_proof : 2⁻¹ + Real.sqrt 16 - (3 - Real.sqrt 3)^0 + |Real.sqrt 2 - 1/2| = 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4128_412884


namespace NUMINAMATH_CALUDE_function_increasing_iff_a_geq_neg_three_l4128_412882

/-- The function f(x) = x^2 + 2(a-1)x + 2 is increasing on [4, +∞) if and only if a ≥ -3 -/
theorem function_increasing_iff_a_geq_neg_three (a : ℝ) :
  (∀ x ≥ 4, Monotone (fun x => x^2 + 2*(a-1)*x + 2)) ↔ a ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_function_increasing_iff_a_geq_neg_three_l4128_412882


namespace NUMINAMATH_CALUDE_f_equals_g_l4128_412837

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 1
def g (t : ℝ) : ℝ := t^2 - 1

-- Theorem stating that f and g are the same function
theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l4128_412837


namespace NUMINAMATH_CALUDE_sin_squared_value_l4128_412832

theorem sin_squared_value (θ : Real) 
  (h : Real.cos θ ^ 4 + Real.sin θ ^ 4 + (Real.cos θ * Real.sin θ) ^ 4 + 
       1 / (Real.cos θ ^ 4 + Real.sin θ ^ 4) = 41 / 16) : 
  Real.sin θ ^ 2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_value_l4128_412832


namespace NUMINAMATH_CALUDE_calculate_expression_largest_integer_solution_three_is_largest_integer_solution_l4128_412830

-- Part 1
theorem calculate_expression : 4 * Real.sin (π / 3) - |-1| + (Real.sqrt 3 - 1)^0 + Real.sqrt 48 = 6 * Real.sqrt 3 := by
  sorry

-- Part 2
theorem largest_integer_solution (x : ℝ) :
  (1/2 * (x - 1) ≤ 1 ∧ 1 - x < 2) → x ≤ 3 := by
  sorry

theorem three_is_largest_integer_solution :
  ∃ (x : ℤ), x = 3 ∧ (1/2 * (x - 1) ≤ 1 ∧ 1 - x < 2) ∧
  ∀ (y : ℤ), y > 3 → ¬(1/2 * (y - 1) ≤ 1 ∧ 1 - y < 2) := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_largest_integer_solution_three_is_largest_integer_solution_l4128_412830


namespace NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l4128_412849

/-- Given 100 pounds of cucumbers with initial 99% water composition by weight,
    prove that after water evaporation resulting in 95% water composition,
    the new weight is 20 pounds. -/
theorem cucumber_weight_after_evaporation
  (initial_weight : ℝ)
  (initial_water_percentage : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_weight = 100)
  (h2 : initial_water_percentage = 0.99)
  (h3 : final_water_percentage = 0.95) :
  let solid_weight := initial_weight * (1 - initial_water_percentage)
  let final_weight := solid_weight / (1 - final_water_percentage)
  final_weight = 20 :=
by sorry

end NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l4128_412849


namespace NUMINAMATH_CALUDE_sanchez_rope_theorem_l4128_412827

/-- The length of rope Mr. Sanchez bought last week in feet -/
def rope_last_week : ℕ := 6

/-- The difference in feet between last week's and this week's rope purchase -/
def rope_difference : ℕ := 4

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The total length of rope Mr. Sanchez bought in inches -/
def total_rope_inches : ℕ := (rope_last_week + (rope_last_week - rope_difference)) * inches_per_foot

theorem sanchez_rope_theorem : total_rope_inches = 96 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_rope_theorem_l4128_412827


namespace NUMINAMATH_CALUDE_product_of_sines_equals_one_fourth_l4128_412824

theorem product_of_sines_equals_one_fourth :
  (1 - Real.sin (π/8)) * (1 - Real.sin (3*π/8)) * (1 - Real.sin (5*π/8)) * (1 - Real.sin (7*π/8)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_equals_one_fourth_l4128_412824


namespace NUMINAMATH_CALUDE_monic_quartic_with_specific_roots_l4128_412829

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 14*x^3 + 57*x^2 - 132*x + 36

-- Theorem statement
theorem monic_quartic_with_specific_roots :
  -- The polynomial is monic
  (∀ x, p x = x^4 - 14*x^3 + 57*x^2 - 132*x + 36) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- 3 + √5 is a root
  p (3 + Real.sqrt 5) = 0 ∧
  -- 4 - √7 is a root
  p (4 - Real.sqrt 7) = 0 :=
sorry

end NUMINAMATH_CALUDE_monic_quartic_with_specific_roots_l4128_412829


namespace NUMINAMATH_CALUDE_marbles_lost_ratio_l4128_412800

/-- Represents the number of marbles Beth has initially -/
def total_marbles : ℕ := 72

/-- Represents the number of colors of marbles -/
def num_colors : ℕ := 3

/-- Represents the number of red marbles lost -/
def red_lost : ℕ := 5

/-- Represents the number of marbles Beth has left after losing some -/
def marbles_left : ℕ := 42

/-- Represents the ratio of yellow marbles lost to red marbles lost -/
def yellow_to_red_ratio : ℕ := 3

theorem marbles_lost_ratio :
  ∃ (blue_lost : ℕ),
    (total_marbles / num_colors = total_marbles / num_colors) ∧
    (total_marbles - red_lost - blue_lost - (yellow_to_red_ratio * red_lost) = marbles_left) ∧
    (blue_lost : ℚ) / red_lost = 2 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_ratio_l4128_412800


namespace NUMINAMATH_CALUDE_vector_dot_product_problem_l4128_412852

theorem vector_dot_product_problem (a b : ℝ × ℝ) (h1 : a = (2, 3)) (h2 : b = (-1, 2)) :
  (a + 2 • b) • b = 14 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_problem_l4128_412852


namespace NUMINAMATH_CALUDE_frozen_food_storage_temp_l4128_412814

def standard_temp : ℝ := -18
def temp_range : ℝ := 2

def is_within_range (temp : ℝ) : Prop :=
  (standard_temp - temp_range) ≤ temp ∧ temp ≤ (standard_temp + temp_range)

theorem frozen_food_storage_temp :
  ¬(is_within_range (-21)) ∧
  is_within_range (-19) ∧
  is_within_range (-18) ∧
  is_within_range (-17) := by
sorry

end NUMINAMATH_CALUDE_frozen_food_storage_temp_l4128_412814


namespace NUMINAMATH_CALUDE_sum_product_difference_l4128_412886

theorem sum_product_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x * y = 96) : 
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_sum_product_difference_l4128_412886


namespace NUMINAMATH_CALUDE_teddys_dogs_l4128_412877

theorem teddys_dogs (teddy_cats ben_cats dave_cats : ℕ) 
  (h1 : teddy_cats = 8)
  (h2 : ben_cats = 0)
  (h3 : dave_cats = teddy_cats + 13)
  (h4 : ∃ (teddy_dogs : ℕ), 
    teddy_dogs + teddy_cats + 
    (teddy_dogs + 9) + ben_cats + 
    (teddy_dogs - 5) + dave_cats = 54) :
  ∃ (teddy_dogs : ℕ), teddy_dogs = 7 := by
sorry

end NUMINAMATH_CALUDE_teddys_dogs_l4128_412877


namespace NUMINAMATH_CALUDE_revenue_change_l4128_412876

theorem revenue_change 
  (original_price original_quantity : ℝ) 
  (price_increase : ℝ) 
  (sales_decrease : ℝ) 
  (h1 : price_increase = 0.5) 
  (h2 : sales_decrease = 0.2) :
  let new_price := original_price * (1 + price_increase)
  let new_quantity := original_quantity * (1 - sales_decrease)
  let original_revenue := original_price * original_quantity
  let new_revenue := new_price * new_quantity
  (new_revenue - original_revenue) / original_revenue = 0.2 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l4128_412876


namespace NUMINAMATH_CALUDE_planes_perpendicular_l4128_412808

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (m n : Line) (α β : Plane) :
  parallel m n → perpendicular n β → subset m α → perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l4128_412808


namespace NUMINAMATH_CALUDE_orange_ring_weight_l4128_412888

/-- The weight of the orange ring in an experiment -/
theorem orange_ring_weight (purple_weight white_weight total_weight : ℚ)
  (h1 : purple_weight = 33/100)
  (h2 : white_weight = 21/50)
  (h3 : total_weight = 83/100) :
  total_weight - (purple_weight + white_weight) = 2/25 := by
  sorry

#eval (83/100 : ℚ) - ((33/100 : ℚ) + (21/50 : ℚ))

end NUMINAMATH_CALUDE_orange_ring_weight_l4128_412888


namespace NUMINAMATH_CALUDE_vector_problem_l4128_412898

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b (y : ℝ) : ℝ × ℝ := (Real.cos y, Real.sin y)
noncomputable def c (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def f (x : ℝ) : ℝ := dot_product (a x) (c x)

noncomputable def g (m x : ℝ) : ℝ := f (x + m)

theorem vector_problem (x y : ℝ) 
  (h : ‖a x - b y‖ = 2 * Real.sqrt 5 / 5) : 
  (Real.cos (x - y) = 3 / 5) ∧ 
  (∃ (m : ℝ), m > 0 ∧ m = Real.pi / 4 ∧ 
    ∀ (n : ℝ), n > 0 → (∀ (t : ℝ), g n t = g n (-t)) → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l4128_412898


namespace NUMINAMATH_CALUDE_unique_solution_l4128_412855

theorem unique_solution : ∃! x : ℝ, 3 * x + 3 * 12 + 3 * 16 + 11 = 134 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l4128_412855


namespace NUMINAMATH_CALUDE_obstacle_course_probability_l4128_412874

def pass_rate_1 : ℝ := 0.8
def pass_rate_2 : ℝ := 0.7
def pass_rate_3 : ℝ := 0.6

theorem obstacle_course_probability :
  let prob_pass_two := pass_rate_1 * pass_rate_2 * (1 - pass_rate_3)
  prob_pass_two = 0.224 := by
sorry

end NUMINAMATH_CALUDE_obstacle_course_probability_l4128_412874


namespace NUMINAMATH_CALUDE_units_digit_of_4_pow_10_l4128_412889

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The statement that the units digit of 4^10 is 6 -/
theorem units_digit_of_4_pow_10 : unitsDigit (4^10) = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_4_pow_10_l4128_412889


namespace NUMINAMATH_CALUDE_jeremy_jerseys_l4128_412804

def jerseyProblem (initialAmount basketballCost shortsCost jerseyCost remainingAmount : ℕ) : Prop :=
  let totalSpent := initialAmount - remainingAmount
  let nonJerseyCost := basketballCost + shortsCost
  let jerseyTotalCost := totalSpent - nonJerseyCost
  jerseyTotalCost / jerseyCost = 5

theorem jeremy_jerseys :
  jerseyProblem 50 18 8 2 14 := by sorry

end NUMINAMATH_CALUDE_jeremy_jerseys_l4128_412804


namespace NUMINAMATH_CALUDE_middle_quad_area_proportion_l4128_412861

-- Define a convex quadrilateral
def ConvexQuadrilateral : Type := Unit

-- Define a function to represent the area of a quadrilateral
def area (q : ConvexQuadrilateral) : ℝ := sorry

-- Define the middle quadrilateral formed by connecting points
def middleQuadrilateral (q : ConvexQuadrilateral) : ConvexQuadrilateral := sorry

-- State the theorem
theorem middle_quad_area_proportion (q : ConvexQuadrilateral) :
  area (middleQuadrilateral q) = (1 / 25) * area q := by sorry

end NUMINAMATH_CALUDE_middle_quad_area_proportion_l4128_412861


namespace NUMINAMATH_CALUDE_fern_purchase_cost_l4128_412806

/-- The total cost of purchasing high heels and ballet slippers -/
def total_cost (high_heel_price : ℝ) (ballet_slipper_ratio : ℝ) (ballet_slipper_count : ℕ) : ℝ :=
  high_heel_price + (ballet_slipper_ratio * high_heel_price * ballet_slipper_count)

/-- Theorem stating the total cost of Fern's purchase -/
theorem fern_purchase_cost :
  total_cost 60 (2/3) 5 = 260 := by
  sorry

end NUMINAMATH_CALUDE_fern_purchase_cost_l4128_412806


namespace NUMINAMATH_CALUDE_polygon_sides_from_exterior_angle_l4128_412831

theorem polygon_sides_from_exterior_angle (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → 
  (exterior_angle > 0) → 
  (exterior_angle < 180) → 
  (n * exterior_angle = 360) → 
  (exterior_angle = 30) → 
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_from_exterior_angle_l4128_412831


namespace NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l4128_412892

-- Define the custom operation
def otimes (a b : ℝ) : ℝ := a^2 - |b|

-- Theorem statement
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l4128_412892


namespace NUMINAMATH_CALUDE_complex_calculation_l4128_412821

theorem complex_calculation (a b : ℂ) (ha : a = 3 + 2*I) (hb : b = 2 - 3*I) :
  3*a + 4*b = 17 - 6*I := by sorry

end NUMINAMATH_CALUDE_complex_calculation_l4128_412821


namespace NUMINAMATH_CALUDE_coin_flip_probability_l4128_412839

theorem coin_flip_probability (n : ℕ) : 
  (1 + n : ℚ) / 2^n = 5/32 ↔ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l4128_412839


namespace NUMINAMATH_CALUDE_regression_properties_l4128_412820

def unit_prices : List ℝ := [4, 5, 6, 7, 8, 9]
def sales_volumes : List ℝ := [90, 84, 83, 80, 75, 68]

def empirical_regression (x : ℝ) (a : ℝ) : ℝ := -4 * x + a

theorem regression_properties :
  let avg_sales := (List.sum sales_volumes) / (List.length sales_volumes)
  let slope := -4
  let a := 106
  (avg_sales = 80) ∧
  (∀ x₁ x₂, empirical_regression x₂ a - empirical_regression x₁ a = slope * (x₂ - x₁)) ∧
  (empirical_regression 10 a = 66) := by
  sorry

end NUMINAMATH_CALUDE_regression_properties_l4128_412820


namespace NUMINAMATH_CALUDE_jori_water_remaining_l4128_412803

/-- The amount of water remaining after usage -/
def water_remaining (initial : ℚ) (usage1 : ℚ) (usage2 : ℚ) : ℚ :=
  initial - usage1 - usage2

/-- Theorem stating the remaining water after Jori's usage -/
theorem jori_water_remaining :
  water_remaining 3 (5/4) (1/2) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_jori_water_remaining_l4128_412803


namespace NUMINAMATH_CALUDE_mikes_work_days_l4128_412865

theorem mikes_work_days (hours_per_day : ℕ) (total_hours : ℕ) (days : ℕ) : 
  hours_per_day = 3 →
  total_hours = 15 →
  days * hours_per_day = total_hours →
  days = 5 := by
sorry

end NUMINAMATH_CALUDE_mikes_work_days_l4128_412865


namespace NUMINAMATH_CALUDE_sqrt_36_minus_k_squared_minus_6_equals_zero_l4128_412838

theorem sqrt_36_minus_k_squared_minus_6_equals_zero (k : ℝ) :
  Real.sqrt (36 - k^2) - 6 = 0 ↔ k = 0 := by sorry

end NUMINAMATH_CALUDE_sqrt_36_minus_k_squared_minus_6_equals_zero_l4128_412838


namespace NUMINAMATH_CALUDE_sqrt_plus_reciprocal_inequality_l4128_412822

theorem sqrt_plus_reciprocal_inequality (x : ℝ) (h : x > 0) :
  Real.sqrt x + 1 / Real.sqrt x ≥ 2 ∧
  (Real.sqrt x + 1 / Real.sqrt x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_reciprocal_inequality_l4128_412822


namespace NUMINAMATH_CALUDE_inverse_38_mod_53_l4128_412825

theorem inverse_38_mod_53 (h : (16⁻¹ : ZMod 53) = 20) : (38⁻¹ : ZMod 53) = 25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_38_mod_53_l4128_412825


namespace NUMINAMATH_CALUDE_statement_relationship_l4128_412859

theorem statement_relationship :
  (∀ x : ℝ, x^2 - 5*x < 0 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ x^2 - 5*x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_statement_relationship_l4128_412859


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_84_and_n_l4128_412880

theorem greatest_common_divisor_of_84_and_n (n : ℕ) : 
  (∃ (d₁ d₂ d₃ : ℕ), d₁ < d₂ ∧ d₂ < d₃ ∧ 
    {d | d > 0 ∧ d ∣ 84 ∧ d ∣ n} = {d₁, d₂, d₃}) →
  (∃ (d : ℕ), d > 0 ∧ d ∣ 84 ∧ d ∣ n ∧ 
    ∀ (k : ℕ), k > 0 ∧ k ∣ 84 ∧ k ∣ n → k ≤ d) →
  4 = (Nat.gcd 84 n) :=
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_84_and_n_l4128_412880


namespace NUMINAMATH_CALUDE_min_value_theorem_l4128_412844

/-- The function f(x) = x|x - a| has a minimum value of 2 on the interval [1, 2] when a = 3 -/
theorem min_value_theorem (a : ℝ) (h1 : a > 0) :
  (∀ x ∈ Set.Icc 1 2, x * |x - a| ≥ 2) ∧ 
  (∃ x ∈ Set.Icc 1 2, x * |x - a| = 2) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4128_412844


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l4128_412891

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 540) (h2 : Nat.gcd p r = 1080) :
  540 ≤ Nat.gcd q r ∧ ∃ (p' q' r' : ℕ+), Nat.gcd p' q' = 540 ∧ Nat.gcd p' r' = 1080 ∧ Nat.gcd q' r' = 540 := by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l4128_412891


namespace NUMINAMATH_CALUDE_evenly_spaced_poles_l4128_412858

/-- Given five evenly spaced poles along a straight road, 
    if the distance between the second and fifth poles is 90 feet, 
    then the distance between the first and fifth poles is 120 feet. -/
theorem evenly_spaced_poles (n : ℕ) (d : ℝ) (h1 : n = 5) (h2 : d = 90) :
  let pole_distance (i j : ℕ) := d * (j - i) / 3
  pole_distance 1 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_evenly_spaced_poles_l4128_412858


namespace NUMINAMATH_CALUDE_fourth_student_score_l4128_412818

theorem fourth_student_score (s1 s2 s3 s4 : ℕ) : 
  s1 = 70 → s2 = 80 → s3 = 90 → (s1 + s2 + s3 + s4) / 4 = 70 → s4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_fourth_student_score_l4128_412818


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_equations_l4128_412851

/-- Definition of an ellipse with given properties -/
def Ellipse (e : ℝ) (d : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (f₁ f₂ : ℝ × ℝ), 
    f₁.2 = 0 ∧ f₂.2 = 0 ∧ 
    (p.1 - f₁.1)^2 + p.2^2 + (p.1 - f₂.1)^2 + p.2^2 = d^2 ∧
    (f₁.1 - f₂.1)^2 = (e * d)^2}

/-- Definition of a hyperbola with given properties -/
def Hyperbola (c : ℝ) (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (f₁ f₂ : ℝ × ℝ),
    f₁.2 = 0 ∧ f₂.2 = 0 ∧ 
    (f₁.1 - f₂.1)^2 = 4 * c^2 ∧
    (p.2 = k * p.1 → p.1^2 * (1 + k^2) = c^2 * (1 + k^2)^2)}

/-- Main theorem statement -/
theorem ellipse_hyperbola_equations :
  ∀ (x y : ℝ),
    (x, y) ∈ Ellipse (1/2) 8 ↔ x^2/16 + y^2/12 = 1 ∧
    (x, y) ∈ Hyperbola 2 (Real.sqrt 3) ↔ x^2 - y^2/3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_equations_l4128_412851


namespace NUMINAMATH_CALUDE_sixth_power_sum_l4128_412857

theorem sixth_power_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^6 + b^6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l4128_412857


namespace NUMINAMATH_CALUDE_calculation_proof_l4128_412823

theorem calculation_proof :
  (3 / (-1/2) - (2/5 - 1/3) * 15 = -7) ∧
  ((-3)^2 - (-2)^3 * (-1/4) - (-1 + 6) = 2) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l4128_412823


namespace NUMINAMATH_CALUDE_perimeter_pedal_relation_not_implies_equilateral_l4128_412809

/-- A triangle with vertices A, B, C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The pedal triangle of a given triangle -/
def pedalTriangle (t : Triangle) : Triangle := sorry

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Theorem stating that the original statement is false -/
theorem perimeter_pedal_relation_not_implies_equilateral :
  ∃ t : Triangle, perimeter t = 2 * perimeter (pedalTriangle t) ∧ ¬isEquilateral t := by
  sorry

end NUMINAMATH_CALUDE_perimeter_pedal_relation_not_implies_equilateral_l4128_412809


namespace NUMINAMATH_CALUDE_second_divisor_l4128_412848

theorem second_divisor (k : ℕ) (h1 : k > 0) (h2 : k < 42) 
  (h3 : k % 5 = 2) (h4 : k % 7 = 3) 
  (d : ℕ) (h5 : d > 0) (h6 : k % d = 5) : d = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_l4128_412848


namespace NUMINAMATH_CALUDE_cubic_function_value_l4128_412860

/-- Given a cubic function f(x) = ax³ + 3 where f(-2) = -5, prove that f(2) = 11 -/
theorem cubic_function_value (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x^3 + 3) 
  (h2 : f (-2) = -5) : f 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_value_l4128_412860


namespace NUMINAMATH_CALUDE_intersection_sum_l4128_412843

theorem intersection_sum (c d : ℝ) :
  (∀ x y : ℝ, x = (1/3) * y + c ↔ y = (1/3) * x + d) →
  3 = (1/3) * 0 + c →
  0 = (1/3) * 3 + d →
  c + d = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l4128_412843


namespace NUMINAMATH_CALUDE_fred_marbles_count_l4128_412845

/-- Represents the number of marbles Fred has of each color -/
structure MarbleCount where
  red : ℕ
  green : ℕ
  dark_blue : ℕ

/-- Calculates the total number of marbles -/
def total_marbles (m : MarbleCount) : ℕ :=
  m.red + m.green + m.dark_blue

/-- Theorem stating the total number of marbles Fred has -/
theorem fred_marbles_count :
  ∃ (m : MarbleCount),
    m.red = 38 ∧
    m.green = m.red / 2 ∧
    m.dark_blue = 6 ∧
    total_marbles m = 63 := by
  sorry

end NUMINAMATH_CALUDE_fred_marbles_count_l4128_412845


namespace NUMINAMATH_CALUDE_sin_585_degrees_l4128_412807

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l4128_412807


namespace NUMINAMATH_CALUDE_digit_sum_congruence_part1_digit_sum_congruence_part2_l4128_412841

/-- S_r(n) is the sum of the digits of n in base r -/
def S_r (r : ℕ) (n : ℕ) : ℕ :=
  sorry

theorem digit_sum_congruence_part1 :
  ∀ r : ℕ, r > 2 → ∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℕ, n > 0 → S_r r n ≡ n [MOD p] :=
sorry

theorem digit_sum_congruence_part2 :
  ∀ r : ℕ, r > 1 → ∀ p : ℕ, Nat.Prime p →
  ∃ f : ℕ → ℕ, Function.Injective f ∧ ∀ k : ℕ, S_r r (f k) ≡ f k [MOD p] :=
sorry

end NUMINAMATH_CALUDE_digit_sum_congruence_part1_digit_sum_congruence_part2_l4128_412841


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4128_412871

/-- Given a quadratic inequality ax² + bx + 2 > 0 with solution set {x | -1/2 < x < 1/3},
    prove that a + b = -14 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  a + b = -14 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4128_412871


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l4128_412840

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 62) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l4128_412840


namespace NUMINAMATH_CALUDE_barbell_cost_l4128_412846

def number_of_barbells : ℕ := 3
def amount_given : ℕ := 850
def change_received : ℕ := 40

theorem barbell_cost :
  (amount_given - change_received) / number_of_barbells = 270 :=
by sorry

end NUMINAMATH_CALUDE_barbell_cost_l4128_412846


namespace NUMINAMATH_CALUDE_block_size_correct_l4128_412873

/-- The number of squares on a standard chessboard -/
def standardChessboardSize : Nat := 64

/-- The number of squares removed from the chessboard -/
def removedSquares : Nat := 2

/-- The number of rectangular blocks that can be placed on the modified chessboard -/
def numberOfBlocks : Nat := 30

/-- The size of the rectangular block in squares -/
def blockSize : Nat := 2

/-- Theorem stating that the given block size is correct for the modified chessboard -/
theorem block_size_correct :
  blockSize * numberOfBlocks ≤ standardChessboardSize - removedSquares ∧
  (blockSize + 1) * numberOfBlocks > standardChessboardSize - removedSquares :=
sorry

end NUMINAMATH_CALUDE_block_size_correct_l4128_412873


namespace NUMINAMATH_CALUDE_imaginary_complex_implies_modulus_l4128_412801

/-- Given a real number t, if the complex number z = (1-ti)/(1+i) is purely imaginary, 
    then |√3 + ti| = 2 -/
theorem imaginary_complex_implies_modulus (t : ℝ) : 
  let z : ℂ := (1 - t * Complex.I) / (1 + Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs (Real.sqrt 3 + t * Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_complex_implies_modulus_l4128_412801


namespace NUMINAMATH_CALUDE_last_locker_is_2046_l4128_412881

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents the corridor of lockers -/
def Corridor := Fin 2048 → LockerState

/-- Represents the student's locker opening strategy -/
def OpeningStrategy := Corridor → Nat → Nat

/-- The final locker opened by the student -/
def lastOpenedLocker (strategy : OpeningStrategy) : Nat :=
  2046

/-- The theorem stating that the last opened locker is 2046 -/
theorem last_locker_is_2046 (strategy : OpeningStrategy) :
  lastOpenedLocker strategy = 2046 := by
  sorry

#check last_locker_is_2046

end NUMINAMATH_CALUDE_last_locker_is_2046_l4128_412881


namespace NUMINAMATH_CALUDE_inverse_function_point_and_sum_l4128_412817

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := sorry

-- State the theorem
theorem inverse_function_point_and_sum :
  (f 2 = 6) →  -- This condition is derived from (2,3) being on y = f(x)/2
  (f_inv 6 = 2) →  -- This is the definition of the inverse function
  (∃ (x y : ℝ), x = 6 ∧ y = 1 ∧ y = (f_inv x) / 2) ∧  -- Point (6,1) is on y = f^(-1)(x)/2
  (6 + 1 = 7)  -- Sum of coordinates
  := by sorry

end NUMINAMATH_CALUDE_inverse_function_point_and_sum_l4128_412817


namespace NUMINAMATH_CALUDE_paul_can_win_2019_paul_cannot_win_2020_l4128_412890

/-- Represents the state of the game at any point --/
structure GameState where
  remaining : Nat  -- Number of marbles remaining
  piles : List Nat -- List of pile sizes

/-- Defines a valid move in the game --/
def validMove (s : GameState) (newPile1 newPile2 : Nat) : Prop :=
  s.remaining > 0 ∧ 
  newPile1 > 0 ∧ 
  newPile2 > 0 ∧ 
  newPile1 + newPile2 = s.remaining - 1

/-- Defines a winning state --/
def isWinningState (s : GameState) : Prop :=
  s.piles.all (· = 3) ∧ s.remaining = 0

/-- Theorem: Paul can win when N = 2019 --/
theorem paul_can_win_2019 : 
  ∃ (strategy : GameState → Nat × Nat), 
    let initialState : GameState := ⟨2019, [2019]⟩
    ∃ (finalState : GameState), 
      (strategy initialState).1 > 0 ∧ 
      (strategy initialState).2 > 0 ∧ 
      (strategy initialState).1 + (strategy initialState).2 = 2018 ∧
      isWinningState finalState :=
sorry

/-- Theorem: Paul cannot win when N = 2020 --/
theorem paul_cannot_win_2020 : 
  ¬∃ (strategy : GameState → Nat × Nat), 
    let initialState : GameState := ⟨2020, [2020]⟩
    ∃ (finalState : GameState), 
      (strategy initialState).1 > 0 ∧ 
      (strategy initialState).2 > 0 ∧ 
      (strategy initialState).1 + (strategy initialState).2 = 2019 ∧
      isWinningState finalState :=
sorry

end NUMINAMATH_CALUDE_paul_can_win_2019_paul_cannot_win_2020_l4128_412890


namespace NUMINAMATH_CALUDE_jeffs_towers_count_l4128_412868

/-- The number of sandcastles on Mark's beach -/
def marks_sandcastles : ℕ := 20

/-- The number of towers per sandcastle on Mark's beach -/
def marks_towers_per_castle : ℕ := 10

/-- The ratio of Jeff's sandcastles to Mark's sandcastles -/
def jeff_to_mark_ratio : ℕ := 3

/-- The total number of sandcastles and towers on both beaches -/
def total_objects : ℕ := 580

/-- The number of towers per sandcastle on Jeff's beach -/
def jeffs_towers_per_castle : ℕ := 5

theorem jeffs_towers_count : jeffs_towers_per_castle = 5 := by
  sorry

end NUMINAMATH_CALUDE_jeffs_towers_count_l4128_412868


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_exists_l4128_412853

theorem rectangular_parallelepiped_exists : ∃ (a b c : ℕ+), 2 * (a * b + b * c + c * a) = 4 * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_exists_l4128_412853


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l4128_412872

theorem diophantine_equation_solutions (n : ℕ+) :
  ∃ (S : Finset (ℤ × ℤ)), S.card ≥ n ∧ ∀ (p : ℤ × ℤ), p ∈ S → p.1^2 + 15 * p.2^2 = 4^(n : ℕ) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l4128_412872


namespace NUMINAMATH_CALUDE_set_B_proof_l4128_412850

def U : Finset Nat := {1,2,3,4,5,6,7,8}

theorem set_B_proof (A B : Finset Nat) 
  (h1 : A ∩ (U \ B) = {1,3})
  (h2 : U \ (A ∪ B) = {2,4}) :
  B = {5,6,7,8} := by
sorry

end NUMINAMATH_CALUDE_set_B_proof_l4128_412850


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l4128_412897

-- Define the line equation
def line_eq (a b x y : ℝ) : Prop := a * x - b * y + 8 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-2, 2)

-- Theorem statement
theorem min_value_of_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_line_passes_center : line_eq a b (circle_center.1) (circle_center.2)) :
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → line_eq a' b' (circle_center.1) (circle_center.2) → 
    1/a + 1/b ≤ 1/a' + 1/b') ∧ 1/a + 1/b = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l4128_412897


namespace NUMINAMATH_CALUDE_carpet_cut_length_l4128_412833

theorem carpet_cut_length (square_area : ℝ) (room_area : ℝ) : 
  square_area = 169 →
  room_area = 143 →
  (Real.sqrt square_area - room_area / Real.sqrt square_area) = 2 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cut_length_l4128_412833


namespace NUMINAMATH_CALUDE_isabel_savings_l4128_412811

def initial_amount : ℚ := 204
def toy_fraction : ℚ := 1/2
def book_fraction : ℚ := 1/2

theorem isabel_savings : 
  initial_amount * (1 - toy_fraction) * (1 - book_fraction) = 51 := by
  sorry

end NUMINAMATH_CALUDE_isabel_savings_l4128_412811


namespace NUMINAMATH_CALUDE_product_expansion_l4128_412842

theorem product_expansion (x : ℝ) : (x + 2) * (x^2 + 3*x + 4) = x^3 + 5*x^2 + 10*x + 8 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l4128_412842


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_six_l4128_412870

theorem infinite_solutions_iff_b_eq_neg_six :
  ∀ b : ℝ, (∀ x : ℝ, 5 * (3 * x - b) = 3 * (5 * x + 10)) ↔ b = -6 := by
sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_six_l4128_412870


namespace NUMINAMATH_CALUDE_elliptical_cylinder_stability_l4128_412895

/-- A cylinder with an elliptical cross-section -/
structure EllipticalCylinder where
  a : ℝ
  b : ℝ
  h : a > b

/-- Stability condition for an elliptical cylinder -/
def is_stable (c : EllipticalCylinder) : Prop :=
  c.b / c.a < 1 / Real.sqrt 2

/-- Theorem: An elliptical cylinder is in stable equilibrium iff b/a < 1/√2 -/
theorem elliptical_cylinder_stability (c : EllipticalCylinder) :
  is_stable c ↔ c.b / c.a < 1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_elliptical_cylinder_stability_l4128_412895


namespace NUMINAMATH_CALUDE_translation_point_difference_l4128_412867

/-- Given points A and B, their translations A₁ and B₁, prove that a - b = -8 -/
theorem translation_point_difference (A B A₁ B₁ : ℝ × ℝ) (a b : ℝ) 
  (h1 : A = (1, -3))
  (h2 : B = (2, 1))
  (h3 : A₁ = (a, 2))
  (h4 : B₁ = (-1, b))
  (h5 : ∃ (v : ℝ × ℝ), A₁ = A + v ∧ B₁ = B + v) :
  a - b = -8 := by
  sorry

end NUMINAMATH_CALUDE_translation_point_difference_l4128_412867


namespace NUMINAMATH_CALUDE_translation_sum_l4128_412869

/-- A translation that moves a point 5 units right and 3 units up -/
def translation (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 5, p.2 + 3)

/-- Apply a translation n times to a point -/
def apply_translation (p : ℝ × ℝ) (n : ℕ) : ℝ × ℝ :=
  Nat.recOn n p (fun _ q => translation q)

theorem translation_sum (initial : ℝ × ℝ) :
  let final := apply_translation initial 6
  final.1 + final.2 = 47 :=
sorry

end NUMINAMATH_CALUDE_translation_sum_l4128_412869


namespace NUMINAMATH_CALUDE_problem_statement_l4128_412847

theorem problem_statement (x : ℂ) (h : x - 1/x = Complex.I * Real.sqrt 3) :
  x^4374 - 1/x^4374 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4128_412847


namespace NUMINAMATH_CALUDE_max_value_a_l4128_412856

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 50) 
  (h5 : d > 10) : 
  a ≤ 2924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 2924 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 50 ∧ 
    d' > 10 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l4128_412856


namespace NUMINAMATH_CALUDE_chess_tournament_games_l4128_412828

theorem chess_tournament_games (n : ℕ) (h : n = 5) : 
  (n * (n - 1)) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l4128_412828


namespace NUMINAMATH_CALUDE_simplify_expression_l4128_412815

theorem simplify_expression (m : ℝ) (h1 : m ≠ 1) (h2 : m ≠ -2) :
  (m^2 - 4*m + 4) / (m - 1) / ((3 / (m - 1)) - m - 1) = (2 - m) / (2 + m) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4128_412815


namespace NUMINAMATH_CALUDE_expression_evaluation_l4128_412826

theorem expression_evaluation : (4^4 - 4*(4-1)^4)^4 = 21381376 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4128_412826


namespace NUMINAMATH_CALUDE_max_value_cubic_function_l4128_412836

theorem max_value_cubic_function (m : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), -y^3 + 6*y^2 - m ≤ -x^3 + 6*x^2 - m) ∧
  (∃ (z : ℝ), -z^3 + 6*z^2 - m = 12) →
  m = 20 := by
sorry

end NUMINAMATH_CALUDE_max_value_cubic_function_l4128_412836


namespace NUMINAMATH_CALUDE_part_one_calculation_part_two_calculation_part_three_calculation_l4128_412885

-- Part 1
theorem part_one_calculation : -12 - (-18) + (-7) = -1 := by sorry

-- Part 2
theorem part_two_calculation : (4/7 - 1/9 + 2/21) * (-63) = -35 := by sorry

-- Part 3
theorem part_three_calculation : (-4)^2 / 2 + 9 * (-1/3) - |3 - 4| = 4 := by sorry

end NUMINAMATH_CALUDE_part_one_calculation_part_two_calculation_part_three_calculation_l4128_412885


namespace NUMINAMATH_CALUDE_intersection_point_l4128_412899

theorem intersection_point (x y : ℚ) :
  (8 * x - 5 * y = 10) ∧ (9 * x + 4 * y = 20) ↔ x = 140 / 77 ∧ y = 70 / 77 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l4128_412899


namespace NUMINAMATH_CALUDE_shirts_made_today_proof_l4128_412878

/-- Calculates the number of shirts made today given the production rate,
    yesterday's working time, and the total number of shirts made. -/
def shirts_made_today (rate : ℕ) (yesterday_time : ℕ) (total_shirts : ℕ) : ℕ :=
  total_shirts - (rate * yesterday_time)

/-- Proves that the number of shirts made today is 84 given the specified conditions. -/
theorem shirts_made_today_proof :
  shirts_made_today 6 12 156 = 84 := by
  sorry

end NUMINAMATH_CALUDE_shirts_made_today_proof_l4128_412878


namespace NUMINAMATH_CALUDE_cube_distance_to_plane_l4128_412835

/-- Given a cube with side length 10 and three vertices adjacent to the closest vertex A
    at heights 10, 11, and 12 above a plane, prove that the distance from A to the plane
    is (33-√294)/3 -/
theorem cube_distance_to_plane (cube_side : ℝ) (height_1 height_2 height_3 : ℝ) :
  cube_side = 10 →
  height_1 = 10 →
  height_2 = 11 →
  height_3 = 12 →
  ∃ (distance : ℝ), distance = (33 - Real.sqrt 294) / 3 ∧
    distance = min height_1 (min height_2 height_3) - 
      Real.sqrt ((cube_side^2 - (height_2 - height_1)^2) / 4 +
                 (cube_side^2 - (height_3 - height_1)^2) / 4 +
                 (cube_side^2 - (height_3 - height_2)^2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_cube_distance_to_plane_l4128_412835


namespace NUMINAMATH_CALUDE_symmetric_about_one_empty_solution_set_implies_a_leq_one_at_most_one_intersection_l4128_412887

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define evenness for a function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Proposition 2
theorem symmetric_about_one (h : is_even (fun x ↦ f (x + 1))) :
  ∀ x, f (1 + x) = f (1 - x) := by sorry

-- Proposition 3
theorem empty_solution_set_implies_a_leq_one (a : ℝ) :
  (∀ x, |x - 4| + |x - 3| ≥ a) → a ≤ 1 := by sorry

-- Proposition 4
theorem at_most_one_intersection (a : ℝ) :
  ∃! y, f a = y := by sorry

end NUMINAMATH_CALUDE_symmetric_about_one_empty_solution_set_implies_a_leq_one_at_most_one_intersection_l4128_412887


namespace NUMINAMATH_CALUDE_danny_carpooling_l4128_412893

/-- Given Danny's carpooling route, prove the distance to the first friend's house -/
theorem danny_carpooling (x : ℝ) :
  x > 0 ∧ 
  (x / 2 > 0) ∧ 
  (3 * (x + x / 2) = 36) →
  x = 8 :=
by sorry

end NUMINAMATH_CALUDE_danny_carpooling_l4128_412893


namespace NUMINAMATH_CALUDE_room_width_calculation_l4128_412864

/-- Given a room with specified length, paving cost, and paving rate, calculate its width -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) 
  (h1 : length = 5.5)
  (h2 : total_cost = 24750)
  (h3 : rate_per_sqm = 1200) :
  total_cost / rate_per_sqm / length = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l4128_412864


namespace NUMINAMATH_CALUDE_cartoon_time_l4128_412879

theorem cartoon_time (cartoon_ratio : ℚ) (chore_ratio : ℚ) (chore_time : ℚ) : 
  cartoon_ratio / chore_ratio = 5 / 4 →
  chore_time = 96 →
  (cartoon_ratio * chore_time) / chore_ratio / 60 = 2 := by
sorry

end NUMINAMATH_CALUDE_cartoon_time_l4128_412879


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l4128_412883

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the property that i^2 = -1
axiom i_squared : i^2 = -1

-- Define the property that powers of i repeat every four powers
axiom i_period (n : ℤ) : i^n = i^(n % 4)

-- State the theorem
theorem sum_of_i_powers : i^23 + i^221 + i^20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l4128_412883


namespace NUMINAMATH_CALUDE_pool_length_is_ten_l4128_412875

/-- Represents a rectangular swimming pool with a surrounding deck -/
structure PoolWithDeck where
  poolLength : ℝ
  poolWidth : ℝ
  deckWidth : ℝ

/-- Calculates the total area of the pool and deck -/
def totalArea (p : PoolWithDeck) : ℝ :=
  (p.poolLength + 2 * p.deckWidth) * (p.poolWidth + 2 * p.deckWidth)

/-- Theorem: The length of the pool is 10 feet given the specified conditions -/
theorem pool_length_is_ten :
  ∃ (p : PoolWithDeck),
    p.poolWidth = 12 ∧
    p.deckWidth = 4 ∧
    totalArea p = 360 ∧
    p.poolLength = 10 := by
  sorry

end NUMINAMATH_CALUDE_pool_length_is_ten_l4128_412875


namespace NUMINAMATH_CALUDE_book_sale_revenue_l4128_412834

/-- Given a collection of books where 2/3 were sold for $3.50 each and 40 remained unsold,
    prove that the total amount received for the sold books is $280. -/
theorem book_sale_revenue (total_books : ℕ) (price_per_book : ℚ) :
  (2 : ℚ) / 3 * total_books + 40 = total_books →
  price_per_book = (7 : ℚ) / 2 →
  ((2 : ℚ) / 3 * total_books) * price_per_book = 280 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l4128_412834


namespace NUMINAMATH_CALUDE_F_opposite_A_l4128_412819

/-- Represents a face of a cube --/
inductive Face : Type
| A | B | C | D | E | F

/-- Represents a cube net that can be folded into a cube --/
structure CubeNet where
  faces : List Face
  can_fold : Bool

/-- Represents a folded cube --/
structure Cube where
  net : CubeNet
  bottom : Face

/-- Defines the opposite face relation in a cube --/
def opposite_face (c : Cube) (f1 f2 : Face) : Prop :=
  f1 ≠ f2 ∧ ∀ (f : Face), f ≠ f1 → f ≠ f2 → (f ∈ c.net.faces)

/-- Theorem: In a cube formed from a net where face F is the bottom, face F is opposite to face A --/
theorem F_opposite_A (c : Cube) (h : c.bottom = Face.F) : opposite_face c Face.A Face.F :=
sorry

end NUMINAMATH_CALUDE_F_opposite_A_l4128_412819


namespace NUMINAMATH_CALUDE_circle_condition_l4128_412863

/-- The equation of a potential circle with parameter a -/
def circle_equation (x y a : ℝ) : ℝ := x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1

/-- The set of a values for which the equation represents a circle -/
def circle_parameter_set : Set ℝ := {a | a < 2 ∨ a > 2}

/-- Theorem stating that the equation represents a circle if and only if a is in the specified set -/
theorem circle_condition (a : ℝ) :
  (∃ h k r : ℝ, r > 0 ∧ ∀ x y : ℝ, circle_equation x y a = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) ↔
  a ∈ circle_parameter_set :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l4128_412863


namespace NUMINAMATH_CALUDE_allocation_methods_l4128_412810

def doctors : ℕ := 2
def nurses : ℕ := 4
def schools : ℕ := 2
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

theorem allocation_methods :
  (Nat.choose doctors doctors_per_school) * (Nat.choose nurses nurses_per_school) = 12 := by
  sorry

end NUMINAMATH_CALUDE_allocation_methods_l4128_412810


namespace NUMINAMATH_CALUDE_exponential_inequality_l4128_412894

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2^a + 2*a = 2^b + 3*b) : a > b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l4128_412894
