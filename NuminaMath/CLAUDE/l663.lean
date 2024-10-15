import Mathlib

namespace NUMINAMATH_CALUDE_positive_solution_x_l663_66311

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y + 3 * x + 2 * y = 12)
  (eq2 : y * z + 5 * y + 3 * z = 15)
  (eq3 : x * z + 5 * x + 4 * z = 40)
  (x_pos : x > 0) : x = 4 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_x_l663_66311


namespace NUMINAMATH_CALUDE_smaller_rectangle_dimensions_l663_66380

theorem smaller_rectangle_dimensions (square_side : ℝ) (small_width : ℝ) :
  square_side = 10 →
  small_width > 0 →
  small_width < square_side →
  small_width * square_side = (1 / 3) * square_side * square_side →
  (small_width, square_side) = (10 / 3, 10) :=
by sorry

end NUMINAMATH_CALUDE_smaller_rectangle_dimensions_l663_66380


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l663_66359

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l663_66359


namespace NUMINAMATH_CALUDE_unit_circle_y_coordinate_l663_66381

theorem unit_circle_y_coordinate 
  (α : Real) 
  (h1 : -3*π/2 < α ∧ α < 0) 
  (h2 : Real.cos (α - π/3) = -Real.sqrt 3 / 3) : 
  ∃ (x₀ y₀ : Real), 
    x₀^2 + y₀^2 = 1 ∧ 
    y₀ = Real.sin α ∧
    y₀ = (-Real.sqrt 6 - 3) / 6 :=
by sorry

end NUMINAMATH_CALUDE_unit_circle_y_coordinate_l663_66381


namespace NUMINAMATH_CALUDE_candy_cost_l663_66385

/-- The cost of candy given initial amounts and final amount after transaction -/
theorem candy_cost (michael_initial : ℕ) (brother_initial : ℕ) (brother_final : ℕ) 
    (h1 : michael_initial = 42)
    (h2 : brother_initial = 17)
    (h3 : brother_final = 35) :
    michael_initial / 2 + brother_initial - brother_final = 3 :=
by sorry

end NUMINAMATH_CALUDE_candy_cost_l663_66385


namespace NUMINAMATH_CALUDE_pythagorean_reciprocal_perimeter_l663_66390

theorem pythagorean_reciprocal_perimeter 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (pythagorean_reciprocal : (a + b) / c = Real.sqrt 2) 
  (area : a * b / 2 = 4) : 
  a + b + c = 4 * Real.sqrt 2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_reciprocal_perimeter_l663_66390


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l663_66316

/-- An arithmetic sequence with given second and eighth terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_proof
  (a : ℕ → ℤ)
  (h_arith : ArithmeticSequence a)
  (h_a2 : a 2 = -6)
  (h_a8 : a 8 = -18) :
  (∃ d : ℤ, d = -2 ∧ ∀ n : ℕ, a n = -2 * n - 2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l663_66316


namespace NUMINAMATH_CALUDE_system_equation_solution_l663_66360

theorem system_equation_solution (m : ℝ) : 
  (∃ x y : ℝ, x - y = m + 2 ∧ x + 3*y = m ∧ x + y = -2) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l663_66360


namespace NUMINAMATH_CALUDE_chairs_count_l663_66333

-- Define the variables
variable (chair_price : ℚ) (table_price : ℚ) (num_chairs : ℕ)

-- Define the conditions
def condition1 (chair_price table_price num_chairs : ℚ) : Prop :=
  num_chairs * chair_price = num_chairs * table_price - 320

def condition2 (chair_price table_price num_chairs : ℚ) : Prop :=
  num_chairs * chair_price = (num_chairs - 5) * table_price

def condition3 (chair_price table_price : ℚ) : Prop :=
  3 * table_price = 5 * chair_price + 48

-- State the theorem
theorem chairs_count 
  (h1 : condition1 chair_price table_price num_chairs)
  (h2 : condition2 chair_price table_price num_chairs)
  (h3 : condition3 chair_price table_price) :
  num_chairs = 20 := by
  sorry

end NUMINAMATH_CALUDE_chairs_count_l663_66333


namespace NUMINAMATH_CALUDE_symmetric_quadratic_l663_66386

/-- A quadratic function f(x) = x² + (a-2)x + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-2)*x + 3

/-- The interval [a, b] -/
def interval (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

/-- The line of symmetry x = 1 -/
def symmetry_line : ℝ := 1

/-- The statement that the graph of f is symmetric about x = 1 on [a, b] -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x ∈ interval a b, f a x = f a (2*symmetry_line - x)

theorem symmetric_quadratic (a b : ℝ) :
  is_symmetric a b → b = 2 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_quadratic_l663_66386


namespace NUMINAMATH_CALUDE_two_month_discount_l663_66396

/-- Calculates the final price of an item after two consecutive percentage discounts --/
theorem two_month_discount (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  initial_price = 1000 ∧ discount1 = 10 ∧ discount2 = 20 →
  initial_price * (1 - discount1 / 100) * (1 - discount2 / 100) = 720 := by
sorry


end NUMINAMATH_CALUDE_two_month_discount_l663_66396


namespace NUMINAMATH_CALUDE_apple_cost_18_pounds_l663_66329

/-- The cost of apples given a specific rate and weight -/
def appleCost (rate : ℚ) (rateWeight : ℚ) (weight : ℚ) : ℚ :=
  (rate * weight) / rateWeight

/-- Theorem: The cost of 18 pounds of apples at a rate of $6 for 6 pounds is $18 -/
theorem apple_cost_18_pounds :
  appleCost 6 6 18 = 18 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_18_pounds_l663_66329


namespace NUMINAMATH_CALUDE_hill_height_l663_66345

/-- The height of a hill given its base depth and proportion to total vertical distance -/
theorem hill_height (base_depth : ℝ) (total_distance : ℝ) 
  (h1 : base_depth = 300)
  (h2 : base_depth = (1/4) * total_distance) : 
  total_distance - base_depth = 900 := by
  sorry

end NUMINAMATH_CALUDE_hill_height_l663_66345


namespace NUMINAMATH_CALUDE_arc_length_sixty_degree_l663_66368

/-- Given a circle with circumference 60 feet and an arc subtended by a central angle of 60°,
    the length of the arc is 10 feet. -/
theorem arc_length_sixty_degree (circle : Real → Real → Prop) 
  (center : Real × Real) (radius : Real) :
  (2 * Real.pi * radius = 60) →  -- Circumference is 60 feet
  (∀ (θ : Real), 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
    circle (center.1 + radius * Real.cos θ) (center.2 + radius * Real.sin θ)) →
  (10 : Real) = (60 / 6) := by sorry

end NUMINAMATH_CALUDE_arc_length_sixty_degree_l663_66368


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l663_66332

/-- Calculates the profit percentage of a dishonest dealer who uses a reduced weight. -/
theorem dishonest_dealer_profit_percentage 
  (claimed_weight : ℝ) 
  (actual_weight : ℝ) 
  (claimed_weight_positive : claimed_weight > 0)
  (actual_weight_positive : actual_weight > 0)
  (actual_weight_less_than_claimed : actual_weight < claimed_weight) :
  (claimed_weight - actual_weight) / actual_weight * 100 = 
  ((1000 - 780) / 780) * 100 :=
by sorry

end NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l663_66332


namespace NUMINAMATH_CALUDE_browser_tabs_l663_66336

theorem browser_tabs (T : ℚ) : 
  (9 / 40 : ℚ) * T = 90 → T = 400 := by
  sorry

end NUMINAMATH_CALUDE_browser_tabs_l663_66336


namespace NUMINAMATH_CALUDE_product_grade_probabilities_l663_66318

theorem product_grade_probabilities :
  ∀ (p_quality p_second : ℝ),
  p_quality = 0.98 →
  p_second = 0.21 →
  0 ≤ p_quality ∧ p_quality ≤ 1 →
  0 ≤ p_second ∧ p_second ≤ 1 →
  ∃ (p_first p_third : ℝ),
    p_first = p_quality - p_second ∧
    p_third = 1 - p_quality ∧
    p_first = 0.77 ∧
    p_third = 0.02 :=
by
  sorry

end NUMINAMATH_CALUDE_product_grade_probabilities_l663_66318


namespace NUMINAMATH_CALUDE_factors_of_8_cube_5_fifth_7_square_l663_66352

def number_of_factors (n : ℕ) : ℕ := sorry

theorem factors_of_8_cube_5_fifth_7_square :
  number_of_factors (8^3 * 5^5 * 7^2) = 180 := by sorry

end NUMINAMATH_CALUDE_factors_of_8_cube_5_fifth_7_square_l663_66352


namespace NUMINAMATH_CALUDE_theta_range_l663_66395

theorem theta_range (θ : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 * Real.cos θ - x*(1-x) + (1-x)^2 * Real.sin θ > 0) →
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12 :=
by sorry

end NUMINAMATH_CALUDE_theta_range_l663_66395


namespace NUMINAMATH_CALUDE_johns_calculation_l663_66378

theorem johns_calculation (y : ℝ) : (y - 15) / 7 = 25 → (y - 7) / 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_johns_calculation_l663_66378


namespace NUMINAMATH_CALUDE_pool_depths_l663_66375

/-- Depths of pools problem -/
theorem pool_depths (john_depth sarah_depth susan_depth : ℝ) : 
  john_depth = 2 * sarah_depth + 5 →
  susan_depth = john_depth + sarah_depth - 3 →
  john_depth = 15 →
  sarah_depth = 5 ∧ susan_depth = 17 := by
  sorry

end NUMINAMATH_CALUDE_pool_depths_l663_66375


namespace NUMINAMATH_CALUDE_f_additive_l663_66350

/-- A function that satisfies f(a+b) = f(a) + f(b) for all real a and b -/
def f (x : ℝ) : ℝ := 3 * x

/-- Theorem stating that f(a+b) = f(a) + f(b) for all real a and b -/
theorem f_additive (a b : ℝ) : f (a + b) = f a + f b := by
  sorry

end NUMINAMATH_CALUDE_f_additive_l663_66350


namespace NUMINAMATH_CALUDE_caco3_decomposition_spontaneity_l663_66343

/-- Represents the thermodynamic properties of a chemical reaction -/
structure ThermodynamicProperties where
  ΔH : ℝ  -- Enthalpy change
  ΔS : ℝ  -- Entropy change

/-- Calculates the Gibbs free energy change for a given temperature -/
def gibbsFreeEnergyChange (props : ThermodynamicProperties) (T : ℝ) : ℝ :=
  props.ΔH - T * props.ΔS

/-- Theorem: For the CaCO₃ decomposition reaction, there exists a temperature
    above which the reaction becomes spontaneous -/
theorem caco3_decomposition_spontaneity 
    (props : ThermodynamicProperties) 
    (h_endothermic : props.ΔH > 0) 
    (h_disorder_increase : props.ΔS > 0) : 
    ∃ T₀ : ℝ, ∀ T > T₀, gibbsFreeEnergyChange props T < 0 := by
  sorry

end NUMINAMATH_CALUDE_caco3_decomposition_spontaneity_l663_66343


namespace NUMINAMATH_CALUDE_xavier_probability_l663_66312

theorem xavier_probability (p_x p_y p_z : ℝ) 
  (h1 : p_y = 1/2)
  (h2 : p_z = 5/8)
  (h3 : p_x * p_y * (1 - p_z) = 0.0375) :
  p_x = 0.2 := by
sorry

end NUMINAMATH_CALUDE_xavier_probability_l663_66312


namespace NUMINAMATH_CALUDE_hyperbola_from_ellipse_foci_l663_66326

/-- Given an ellipse with equation x²/4 + y² = 1, prove that the hyperbola 
    with equation x²/2 - y² = 1 shares the same foci as the ellipse and 
    passes through the point (2,1) -/
theorem hyperbola_from_ellipse_foci (x y : ℝ) : 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
   (x^2 / (4 : ℝ) + y^2 = 1) ∧ 
   (c^2 = a^2 + b^2) ∧
   (a^2 = 2) ∧ 
   (b^2 = 1) ∧ 
   (c^2 = 3)) →
  (x^2 / (2 : ℝ) - y^2 = 1) ∧ 
  ((2 : ℝ)^2 / (2 : ℝ) - 1^2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_from_ellipse_foci_l663_66326


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l663_66324

def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x^2 + 3 * x + k^2 - 4

theorem unique_solution_quadratic (k : ℝ) :
  (quadratic_equation k 0 = 0) →
  (∃! x, quadratic_equation k x = 0) →
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l663_66324


namespace NUMINAMATH_CALUDE_remainder_seven_205_mod_12_l663_66327

theorem remainder_seven_205_mod_12 : 7^205 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_205_mod_12_l663_66327


namespace NUMINAMATH_CALUDE_next_two_numbers_after_one_l663_66399

def square_sum (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def satisfies_condition (n : ℕ) : Prop :=
  is_perfect_square (square_sum n / n)

theorem next_two_numbers_after_one (n : ℕ) : 
  (n > 1 ∧ n < 337 → ¬satisfies_condition n) ∧
  satisfies_condition 337 ∧
  (n > 337 ∧ n < 65521 → ¬satisfies_condition n) ∧
  satisfies_condition 65521 :=
sorry

end NUMINAMATH_CALUDE_next_two_numbers_after_one_l663_66399


namespace NUMINAMATH_CALUDE_tank_filling_time_l663_66364

theorem tank_filling_time (fast_rate slow_rate : ℝ) (combined_time : ℝ) : 
  fast_rate = 4 * slow_rate →
  1 / combined_time = fast_rate + slow_rate →
  combined_time = 40 →
  1 / slow_rate = 200 := by
sorry

end NUMINAMATH_CALUDE_tank_filling_time_l663_66364


namespace NUMINAMATH_CALUDE_multiplication_fraction_simplification_l663_66305

theorem multiplication_fraction_simplification :
  8 * (2 / 17) * 34 * (1 / 4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_fraction_simplification_l663_66305


namespace NUMINAMATH_CALUDE_rectangle_area_l663_66309

theorem rectangle_area (perimeter : ℝ) (length_width_ratio : ℝ) : 
  perimeter = 60 → length_width_ratio = 1.5 → 
  let width := perimeter / (2 * (1 + length_width_ratio))
  let length := length_width_ratio * width
  let area := length * width
  area = 216 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l663_66309


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_equals_five_l663_66315

theorem x_squared_minus_y_squared_equals_five
  (a : ℝ) (x y : ℝ) (h1 : a^x * a^y = a^5) (h2 : a^x / a^y = a) :
  x^2 - y^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_equals_five_l663_66315


namespace NUMINAMATH_CALUDE_max_score_is_31_l663_66354

/-- Represents a problem-solving robot with a limited IQ balance. -/
structure Robot where
  iq : ℕ

/-- Represents a problem with a score. -/
structure Problem where
  score : ℕ

/-- Calculates the maximum achievable score for a robot solving a set of problems. -/
def maxAchievableScore (initialIQ : ℕ) (problems : List Problem) : ℕ :=
  sorry

/-- The theorem stating the maximum achievable score for the given conditions. -/
theorem max_score_is_31 :
  let initialIQ := 25
  let problems := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map Problem.mk
  maxAchievableScore initialIQ problems = 31 := by
  sorry

end NUMINAMATH_CALUDE_max_score_is_31_l663_66354


namespace NUMINAMATH_CALUDE_unique_solution_set_l663_66367

-- Define the set A
def A : Set ℝ := {a | ∃! x, (x^2 - 4) / (x + a) = 1}

-- Theorem statement
theorem unique_solution_set : A = {-17/4, -2, 2} := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_set_l663_66367


namespace NUMINAMATH_CALUDE_equal_apple_distribution_l663_66356

theorem equal_apple_distribution (total_apples : Nat) (num_students : Nat) 
  (h1 : total_apples = 360) (h2 : num_students = 60) :
  total_apples / num_students = 6 := by
  sorry

end NUMINAMATH_CALUDE_equal_apple_distribution_l663_66356


namespace NUMINAMATH_CALUDE_f_min_correct_l663_66347

noncomputable section

/-- The function f(x) = x^2 - 4x + (2-a)ln(x) where a ∈ ℝ and a ≠ 0 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + (2-a)*Real.log x

/-- The minimum value of f(x) on the interval [e, e^2] -/
def f_min (a : ℝ) : ℝ :=
  if a ≥ 2*(Real.exp 2 - 1)^2 then
    Real.exp 4 - 4*Real.exp 2 + 4 - 2*a
  else if 2*(Real.exp 1 - 1)^2 < a ∧ a < 2*(Real.exp 2 - 1)^2 then
    a/2 - Real.sqrt (2*a) - 3 + (2-a)*Real.log (1 + Real.sqrt (2*a)/2)
  else
    Real.exp 2 - 4*Real.exp 1 + 2 - a

theorem f_min_correct (a : ℝ) (h : a ≠ 0) :
  ∀ x ∈ Set.Icc (Real.exp 1) (Real.exp 2), f_min a ≤ f a x :=
sorry

end

end NUMINAMATH_CALUDE_f_min_correct_l663_66347


namespace NUMINAMATH_CALUDE_system_solution_l663_66325

theorem system_solution (x y : ℝ) : 
  x^3 - x + 1 = y^2 ∧ y^3 - y + 1 = x^2 → 
  (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l663_66325


namespace NUMINAMATH_CALUDE_sum_of_cubes_l663_66339

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l663_66339


namespace NUMINAMATH_CALUDE_candle_box_cost_l663_66306

/-- The cost of a box of candles --/
def box_cost : ℕ := 5

/-- Kerry's age --/
def kerry_age : ℕ := 8

/-- Number of cakes Kerry wants --/
def num_cakes : ℕ := 3

/-- Number of candles in a box --/
def candles_per_box : ℕ := 12

/-- Total number of candles needed --/
def total_candles : ℕ := num_cakes * kerry_age

theorem candle_box_cost : box_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_candle_box_cost_l663_66306


namespace NUMINAMATH_CALUDE_jellybean_problem_l663_66365

theorem jellybean_problem (initial_quantity : ℕ) : 
  (initial_quantity : ℝ) * (0.75^3) = 27 → initial_quantity = 64 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l663_66365


namespace NUMINAMATH_CALUDE_angle4_value_l663_66303

-- Define the angles
def angle1 : ℝ := sorry
def angle2 : ℝ := sorry
def angle3 : ℝ := sorry
def angle4 : ℝ := sorry
def angleA : ℝ := 80
def angleB : ℝ := 50

-- State the theorem
theorem angle4_value :
  (angle1 + angle2 = 180) →
  (angle3 = angle4) →
  (angle1 + angleA + angleB = 180) →
  (angle2 + angle3 + angle4 = 180) →
  angle4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_angle4_value_l663_66303


namespace NUMINAMATH_CALUDE_omelet_preparation_time_l663_66307

/-- Calculates the total time spent preparing and cooking omelets -/
def total_omelet_time (pepper_time onion_time mushroom_time tomato_time cheese_time cook_time : ℕ)
                      (num_peppers num_onions num_mushrooms num_tomatoes num_omelets : ℕ) : ℕ :=
  pepper_time * num_peppers +
  onion_time * num_onions +
  mushroom_time * num_mushrooms +
  tomato_time * num_tomatoes +
  cheese_time * num_omelets +
  cook_time * num_omelets

/-- Proves that the total time spent preparing and cooking 10 omelets is 140 minutes -/
theorem omelet_preparation_time :
  total_omelet_time 3 4 2 3 1 6 8 4 6 6 10 = 140 := by
  sorry

end NUMINAMATH_CALUDE_omelet_preparation_time_l663_66307


namespace NUMINAMATH_CALUDE_max_temperature_range_l663_66363

/-- Given weather conditions and temperatures, calculate the maximum temperature range --/
theorem max_temperature_range 
  (avg_temp : ℝ) 
  (lowest_temp : ℝ) 
  (temp_fluctuation : ℝ) 
  (h1 : avg_temp = 50)
  (h2 : lowest_temp = 45)
  (h3 : temp_fluctuation = 5) :
  (avg_temp + temp_fluctuation) - lowest_temp = 10 := by
sorry

end NUMINAMATH_CALUDE_max_temperature_range_l663_66363


namespace NUMINAMATH_CALUDE_square_root_sum_equals_five_l663_66383

theorem square_root_sum_equals_five : 
  Real.sqrt ((5 / 2 - 3 * Real.sqrt 3 / 2) ^ 2) + Real.sqrt ((5 / 2 + 3 * Real.sqrt 3 / 2) ^ 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_five_l663_66383


namespace NUMINAMATH_CALUDE_female_officers_count_l663_66322

theorem female_officers_count (total_on_duty : ℕ) (female_percentage : ℚ) 
  (h1 : total_on_duty = 180)
  (h2 : female_percentage = 18 / 100)
  (h3 : (total_on_duty / 2 : ℚ) = female_percentage * (female_officers_total : ℚ)) :
  female_officers_total = 500 :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l663_66322


namespace NUMINAMATH_CALUDE_book_purchase_total_price_l663_66317

theorem book_purchase_total_price : 
  let total_books : ℕ := 90
  let math_books : ℕ := 53
  let math_book_price : ℕ := 4
  let history_book_price : ℕ := 5
  let history_books : ℕ := total_books - math_books
  let total_price : ℕ := math_books * math_book_price + history_books * history_book_price
  total_price = 397 := by
sorry

end NUMINAMATH_CALUDE_book_purchase_total_price_l663_66317


namespace NUMINAMATH_CALUDE_geometric_sequence_a9_l663_66372

def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a9 (a : ℕ → ℤ) :
  is_geometric_sequence a →
  a 2 * a 5 = -32 →
  a 3 + a 4 = 4 →
  (∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q) →
  a 9 = -256 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a9_l663_66372


namespace NUMINAMATH_CALUDE_average_of_rst_l663_66353

theorem average_of_rst (r s t : ℝ) (h : (4 / 3) * (r + s + t) = 12) : 
  (r + s + t) / 3 = 3 := by sorry

end NUMINAMATH_CALUDE_average_of_rst_l663_66353


namespace NUMINAMATH_CALUDE_point_on_decreasing_linear_function_l663_66301

/-- A linear function that decreases as x increases -/
def decreasingLinearFunction (k : ℝ) (x : ℝ) : ℝ :=
  k * (x - 2) + 4

/-- The slope of the linear function is negative -/
def isDecreasing (k : ℝ) : Prop :=
  k < 0

/-- The point (3, -1) lies on the graph of the function -/
def pointOnGraph (k : ℝ) : Prop :=
  decreasingLinearFunction k 3 = -1

/-- Theorem: If the linear function y = k(x-2) + 4 is decreasing,
    then the point (3, -1) lies on its graph -/
theorem point_on_decreasing_linear_function :
  ∀ k : ℝ, isDecreasing k → pointOnGraph k :=
by
  sorry

end NUMINAMATH_CALUDE_point_on_decreasing_linear_function_l663_66301


namespace NUMINAMATH_CALUDE_balanced_quadruple_inequality_l663_66344

def balanced (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem balanced_quadruple_inequality (x : ℝ) :
  (∀ a b c d : ℝ, balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔
  x ≥ 3/2 := by sorry

end NUMINAMATH_CALUDE_balanced_quadruple_inequality_l663_66344


namespace NUMINAMATH_CALUDE_max_l_shapes_in_grid_l663_66389

/-- Represents a 6x6 grid --/
def Grid := Fin 6 → Fin 6 → Bool

/-- An L-shape tetromino --/
structure LShape :=
  (position : Fin 6 × Fin 6)
  (orientation : Fin 4)

/-- Checks if an L-shape is within the grid bounds --/
def isWithinBounds (l : LShape) : Bool :=
  sorry

/-- Checks if two L-shapes overlap --/
def doOverlap (l1 l2 : LShape) : Bool :=
  sorry

/-- Checks if a set of L-shapes is valid (within bounds and non-overlapping) --/
def isValidPlacement (shapes : List LShape) : Bool :=
  sorry

/-- The main theorem stating the maximum number of L-shapes in a 6x6 grid --/
theorem max_l_shapes_in_grid :
  ∃ (shapes : List LShape),
    shapes.length = 4 ∧
    isValidPlacement shapes ∧
    ∀ (other_shapes : List LShape),
      isValidPlacement other_shapes →
      other_shapes.length ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_l_shapes_in_grid_l663_66389


namespace NUMINAMATH_CALUDE_root_relationship_l663_66371

/-- Given two functions f and g, and their respective roots x₁ and x₂, prove that x₁ < x₂ -/
theorem root_relationship (f g : ℝ → ℝ) (x₁ x₂ : ℝ) :
  (f = λ x => x + 2^x) →
  (g = λ x => x + Real.log x) →
  f x₁ = 0 →
  g x₂ = 0 →
  x₁ < x₂ := by
  sorry

end NUMINAMATH_CALUDE_root_relationship_l663_66371


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l663_66323

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.05 * L) (h2 : B' * L' = 1.2075 * (B * L)) : B' = 1.15 * B := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l663_66323


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l663_66348

theorem unique_solution_for_equation : ∃! (x y : ℕ), 
  x < 10 ∧ y < 10 ∧ (10 + x) * (200 + 10 * y + 7) = 5166 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l663_66348


namespace NUMINAMATH_CALUDE_shortest_path_length_l663_66369

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 7.5)^2 + (y - 10)^2 = 36
def circle2 (x y : ℝ) : Prop := (x - 15)^2 + (y - 5)^2 = 16

-- Define a path that avoids the circles
def valid_path (p : ℝ → ℝ × ℝ) : Prop :=
  (p 0 = (0, 0)) ∧ 
  (p 1 = (15, 20)) ∧ 
  ∀ t ∈ (Set.Icc 0 1), ¬(circle1 (p t).1 (p t).2) ∧ ¬(circle2 (p t).1 (p t).2)

-- Define the length of a path
def path_length (p : ℝ → ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem shortest_path_length :
  ∃ p, valid_path p ∧ 
    path_length p = 30.6 + 5 * Real.pi / 3 ∧
    ∀ q, valid_path q → path_length p ≤ path_length q :=
sorry

end NUMINAMATH_CALUDE_shortest_path_length_l663_66369


namespace NUMINAMATH_CALUDE_simplify_expression_l663_66351

theorem simplify_expression (x : ℝ) : (3*x - 6)*(x + 9) - (x + 6)*(3*x + 2) = x - 66 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l663_66351


namespace NUMINAMATH_CALUDE_line_slope_one_m_value_l663_66387

/-- Given a line passing through points P(-2, m) and Q(m, 4) with a slope of 1, prove that m = 1 -/
theorem line_slope_one_m_value (m : ℝ) : 
  (4 - m) / (m + 2) = 1 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_one_m_value_l663_66387


namespace NUMINAMATH_CALUDE_marble_selection_problem_l663_66337

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem marble_selection_problem :
  let total_marbles : ℕ := 15
  let required_marbles : ℕ := 2
  let marbles_to_choose : ℕ := 5
  let remaining_marbles : ℕ := total_marbles - required_marbles
  let additional_marbles : ℕ := marbles_to_choose - required_marbles
  choose remaining_marbles additional_marbles = 286 :=
by sorry

end NUMINAMATH_CALUDE_marble_selection_problem_l663_66337


namespace NUMINAMATH_CALUDE_pen_transaction_profit_l663_66379

/-- Calculates the profit percentage for a given transaction -/
def profit_percent (items_bought : ℕ) (price_paid : ℕ) (discount_percent : ℚ) : ℚ :=
  let cost_per_item : ℚ := price_paid / items_bought
  let selling_price_per_item : ℚ := 1 - (discount_percent / 100)
  let total_revenue : ℚ := items_bought * selling_price_per_item
  let profit : ℚ := total_revenue - price_paid
  (profit / price_paid) * 100

/-- The profit percent for the given transaction is approximately 20.52% -/
theorem pen_transaction_profit :
  ∃ ε > 0, |profit_percent 56 46 1 - 20.52| < ε :=
sorry

end NUMINAMATH_CALUDE_pen_transaction_profit_l663_66379


namespace NUMINAMATH_CALUDE_binomial_18_10_l663_66328

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 42328 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l663_66328


namespace NUMINAMATH_CALUDE_banana_cream_pie_angle_l663_66361

def total_students : ℕ := 48
def chocolate_preference : ℕ := 15
def apple_preference : ℕ := 9
def blueberry_preference : ℕ := 11

def remaining_students : ℕ := total_students - (chocolate_preference + apple_preference + blueberry_preference)

def banana_cream_preference : ℕ := remaining_students / 2

theorem banana_cream_pie_angle :
  (banana_cream_preference : ℝ) / total_students * 360 = 45 := by
  sorry

end NUMINAMATH_CALUDE_banana_cream_pie_angle_l663_66361


namespace NUMINAMATH_CALUDE_weight_placement_theorem_l663_66382

def factorial_double (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => (2 * k + 1) * factorial_double k

def weight_placement_ways (n : ℕ) : ℕ :=
  factorial_double n

theorem weight_placement_theorem (n : ℕ) (h : n > 0) :
  weight_placement_ways n = factorial_double n :=
by
  sorry

end NUMINAMATH_CALUDE_weight_placement_theorem_l663_66382


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l663_66393

theorem sum_of_reciprocals_squared (a b c d : ℝ) :
  a = 2 * Real.sqrt 2 + 2 * Real.sqrt 3 + 2 * Real.sqrt 5 →
  b = -2 * Real.sqrt 2 + 2 * Real.sqrt 3 + 2 * Real.sqrt 5 →
  c = 2 * Real.sqrt 2 - 2 * Real.sqrt 3 + 2 * Real.sqrt 5 →
  d = -2 * Real.sqrt 2 - 2 * Real.sqrt 3 + 2 * Real.sqrt 5 →
  (1/a + 1/b + 1/c + 1/d)^2 = 4/45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l663_66393


namespace NUMINAMATH_CALUDE_vectors_form_basis_l663_66384

def v1 : Fin 2 → ℝ := ![2, 3]
def v2 : Fin 2 → ℝ := ![-4, 6]

theorem vectors_form_basis : LinearIndependent ℝ ![v1, v2] :=
sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l663_66384


namespace NUMINAMATH_CALUDE_martin_oranges_l663_66357

/-- Represents the number of fruits Martin has initially -/
def initial_fruits : ℕ := 150

/-- Represents the number of oranges Martin has after eating half of his fruits -/
def oranges : ℕ := 50

/-- Represents the number of limes Martin has after eating half of his fruits -/
def limes : ℕ := 25

/-- Proves that Martin has 50 oranges after eating half of his fruits -/
theorem martin_oranges :
  (oranges + limes = initial_fruits / 2) ∧
  (oranges = 2 * limes) ∧
  (oranges = 50) :=
sorry

end NUMINAMATH_CALUDE_martin_oranges_l663_66357


namespace NUMINAMATH_CALUDE_perfume_tax_rate_l663_66319

theorem perfume_tax_rate (price_before_tax : ℝ) (total_price : ℝ) : price_before_tax = 92 → total_price = 98.90 → (total_price - price_before_tax) / price_before_tax * 100 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_perfume_tax_rate_l663_66319


namespace NUMINAMATH_CALUDE_circle_m_range_and_perpendicular_intersection_l663_66377

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_m_range_and_perpendicular_intersection :
  -- Part 1: If the equation represents a circle, then m ∈ (-∞, 5)
  (∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → m < 5) ∧
  -- Part 2: If the circle intersects the line and OM ⟂ ON, then m = 8/5
  (∀ m : ℝ, 
    (∃ x1 y1 x2 y2 : ℝ, 
      circle_equation x1 y1 m ∧ 
      circle_equation x2 y2 m ∧
      line_equation x1 y1 ∧ 
      line_equation x2 y2 ∧
      perpendicular x1 y1 x2 y2) → 
    m = 8/5) :=
by sorry

end NUMINAMATH_CALUDE_circle_m_range_and_perpendicular_intersection_l663_66377


namespace NUMINAMATH_CALUDE_books_returned_on_wednesday_l663_66392

theorem books_returned_on_wednesday (initial_books : ℕ) (tuesday_out : ℕ) (thursday_out : ℕ) (final_books : ℕ) : 
  initial_books = 250 → 
  tuesday_out = 120 → 
  thursday_out = 15 → 
  final_books = 150 → 
  initial_books - tuesday_out + (initial_books - tuesday_out - final_books + thursday_out) - thursday_out = final_books := by
  sorry

end NUMINAMATH_CALUDE_books_returned_on_wednesday_l663_66392


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l663_66346

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 → 1/a < 1) ∧ (∃ a, 1/a < 1 ∧ a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l663_66346


namespace NUMINAMATH_CALUDE_inequality_solution_l663_66335

theorem inequality_solution (x : ℝ) : 2 * (2 * x - 1) > 3 * x - 1 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l663_66335


namespace NUMINAMATH_CALUDE_spade_or_club_probability_l663_66302

/-- A standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)

/-- The probability of drawing a card of a specific type from a deck -/
def draw_probability (deck : Deck) (favorable_cards : ℕ) : ℚ :=
  favorable_cards / deck.total_cards

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    ranks := 13,
    suits := 4 }

/-- Theorem: The probability of drawing either a ♠ or a ♣ from a standard 52-card deck is 1/2 -/
theorem spade_or_club_probability :
  draw_probability standard_deck (2 * standard_deck.ranks) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_spade_or_club_probability_l663_66302


namespace NUMINAMATH_CALUDE_scooter_price_l663_66330

-- Define the upfront payment and the percentage paid
def upfront_payment : ℝ := 240
def percentage_paid : ℝ := 20

-- State the theorem
theorem scooter_price : 
  (upfront_payment / (percentage_paid / 100)) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_scooter_price_l663_66330


namespace NUMINAMATH_CALUDE_canada_population_1998_l663_66338

theorem canada_population_1998 : 
  (30.3 : ℝ) * 1000000 = 30300000 := by
  sorry

end NUMINAMATH_CALUDE_canada_population_1998_l663_66338


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l663_66388

theorem max_tiles_on_floor (floor_length floor_width tile_length tile_width : ℕ) 
  (h1 : floor_length = 120) 
  (h2 : floor_width = 150) 
  (h3 : tile_length = 50) 
  (h4 : tile_width = 40) : 
  (max 
    ((floor_length / tile_length) * (floor_width / tile_width))
    ((floor_length / tile_width) * (floor_width / tile_length))) = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l663_66388


namespace NUMINAMATH_CALUDE_cricketer_average_score_l663_66321

theorem cricketer_average_score (total_matches : ℕ) (first_set : ℕ) (second_set : ℕ)
  (avg_first : ℚ) (avg_second : ℚ) :
  total_matches = first_set + second_set →
  first_set = 2 →
  second_set = 3 →
  avg_first = 40 →
  avg_second = 10 →
  (first_set * avg_first + second_set * avg_second) / total_matches = 22 :=
by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l663_66321


namespace NUMINAMATH_CALUDE_concert_ticket_revenue_l663_66308

/-- Calculates the total revenue from concert ticket sales given specific discount conditions --/
theorem concert_ticket_revenue :
  let regular_price : ℚ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let total_customers : ℕ := 50
  let first_discount : ℚ := 0.4
  let second_discount : ℚ := 0.15
  
  let first_group_revenue := first_group_size * (regular_price * (1 - first_discount))
  let second_group_revenue := second_group_size * (regular_price * (1 - second_discount))
  let remaining_customers := total_customers - first_group_size - second_group_size
  let full_price_revenue := remaining_customers * regular_price
  
  let total_revenue := first_group_revenue + second_group_revenue + full_price_revenue
  
  total_revenue = 860 := by sorry

end NUMINAMATH_CALUDE_concert_ticket_revenue_l663_66308


namespace NUMINAMATH_CALUDE_emmy_lost_ipods_l663_66391

/-- The number of iPods Emmy lost -/
def ipods_lost : ℕ := sorry

/-- The number of iPods Rosa has -/
def rosa_ipods : ℕ := sorry

theorem emmy_lost_ipods : ipods_lost = 6 :=
  by
  have h1 : 14 - ipods_lost = 2 * rosa_ipods := sorry
  have h2 : (14 - ipods_lost) + rosa_ipods = 12 := sorry
  sorry

#check emmy_lost_ipods

end NUMINAMATH_CALUDE_emmy_lost_ipods_l663_66391


namespace NUMINAMATH_CALUDE_trig_product_equals_one_l663_66334

theorem trig_product_equals_one : 
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
sorry

end NUMINAMATH_CALUDE_trig_product_equals_one_l663_66334


namespace NUMINAMATH_CALUDE_total_balloons_l663_66340

theorem total_balloons (tom_balloons sara_balloons : ℕ) 
  (h1 : tom_balloons = 9) 
  (h2 : sara_balloons = 8) : 
  tom_balloons + sara_balloons = 17 := by
sorry

end NUMINAMATH_CALUDE_total_balloons_l663_66340


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l663_66300

theorem smallest_number_divisible (n : ℕ) : n = 84 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 24 = 5 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 24 = 10 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 24 = 15 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 24 = 20 * k)) ∧ 
  (∃ k1 k2 k3 k4 : ℕ, 
    n - 24 = 5 * k1 ∧ 
    n - 24 = 10 * k2 ∧ 
    n - 24 = 15 * k3 ∧ 
    n - 24 = 20 * k4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l663_66300


namespace NUMINAMATH_CALUDE_smallest_prime_8_less_than_square_l663_66362

theorem smallest_prime_8_less_than_square : 
  ∃ (n : ℕ), 17 = n^2 - 8 ∧ 
  Prime 17 ∧ 
  ∀ (m : ℕ) (p : ℕ), m < n → p = m^2 - 8 → p ≤ 0 ∨ ¬ Prime p :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_8_less_than_square_l663_66362


namespace NUMINAMATH_CALUDE_probability_non_defective_pencils_l663_66370

theorem probability_non_defective_pencils :
  let total_pencils : ℕ := 8
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let total_combinations : ℕ := Nat.choose total_pencils selected_pencils
  let non_defective_combinations : ℕ := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations : ℚ) / total_combinations = 5 / 14 :=
by sorry

end NUMINAMATH_CALUDE_probability_non_defective_pencils_l663_66370


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l663_66398

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 4 + a 7 = 39 →
  a 2 + a 5 + a 8 = 33 →
  a 5 + a 8 + a 11 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l663_66398


namespace NUMINAMATH_CALUDE_max_cash_prize_value_l663_66313

/-- Represents the promotion setup in a shopping mall -/
structure PromotionSetup where
  total_items : Nat
  daily_necessities : Nat
  chosen_items : Nat
  price_increase : ℝ
  lottery_chances : Nat
  win_probability : ℝ

/-- Calculates the expected value of the total cash prize -/
def expected_cash_prize (m : ℝ) (setup : PromotionSetup) : ℝ :=
  setup.lottery_chances * setup.win_probability * m

/-- Theorem stating the maximum value of m for an advantageous promotion -/
theorem max_cash_prize_value (setup : PromotionSetup) :
  setup.total_items = 7 →
  setup.daily_necessities = 3 →
  setup.chosen_items = 3 →
  setup.price_increase = 150 →
  setup.lottery_chances = 3 →
  setup.win_probability = 1/2 →
  ∃ (m : ℝ), m = 100 ∧ 
    ∀ (x : ℝ), expected_cash_prize x setup ≤ setup.price_increase → x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_cash_prize_value_l663_66313


namespace NUMINAMATH_CALUDE_remaining_weight_calculation_l663_66320

/-- Calculates the total remaining weight of groceries after an accident --/
theorem remaining_weight_calculation (green_beans_weight : ℝ) : 
  green_beans_weight = 60 →
  let rice_weight := green_beans_weight - 30
  let sugar_weight := green_beans_weight - 10
  let rice_remaining := rice_weight * (2/3)
  let sugar_remaining := sugar_weight * (4/5)
  rice_remaining + sugar_remaining + green_beans_weight = 120 :=
by
  sorry


end NUMINAMATH_CALUDE_remaining_weight_calculation_l663_66320


namespace NUMINAMATH_CALUDE_slower_train_speed_l663_66394

/-- Proves that the speed of the slower train is 36 kmph given the conditions of the problem -/
theorem slower_train_speed 
  (faster_speed : ℝ) 
  (faster_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : faster_speed = 72) 
  (h2 : faster_length = 180) 
  (h3 : crossing_time = 18) : 
  ∃ (slower_speed : ℝ), slower_speed = 36 ∧ 
    faster_length = (faster_speed - slower_speed) * (5/18) * crossing_time :=
sorry

end NUMINAMATH_CALUDE_slower_train_speed_l663_66394


namespace NUMINAMATH_CALUDE_elise_savings_elise_savings_proof_l663_66314

/-- Proves that Elise saved $13 from her allowance -/
theorem elise_savings : ℕ → Prop :=
  fun (saved : ℕ) =>
    let initial : ℕ := 8
    let comic_cost : ℕ := 2
    let puzzle_cost : ℕ := 18
    let final : ℕ := 1
    initial + saved - (comic_cost + puzzle_cost) = final →
    saved = 13

/-- The proof of the theorem -/
theorem elise_savings_proof : elise_savings 13 := by
  sorry

end NUMINAMATH_CALUDE_elise_savings_elise_savings_proof_l663_66314


namespace NUMINAMATH_CALUDE_problem_solution_l663_66374

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x) * Real.log x + a*x^2 + 2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x - 2

theorem problem_solution :
  (∀ x : ℝ, x > 0 → 3*x + (f (-1) x) - 4 = 0) ∧
  (∀ a : ℝ, a > 0 → (∃! x : ℝ, g a x = 0) → a = 1) ∧
  (∀ x : ℝ, Real.exp (-2) < x → x < Real.exp 1 → g 1 x ≤ 2 * Real.exp 2 - 3 * Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l663_66374


namespace NUMINAMATH_CALUDE_average_timing_error_l663_66373

def total_watches : ℕ := 10

def timing_errors : List ℕ := [0, 1, 2, 3]
def error_frequencies : List ℕ := [3, 4, 2, 1]

def average_error : ℚ := 1.1

theorem average_timing_error :
  (List.sum (List.zipWith (· * ·) timing_errors error_frequencies) : ℚ) / total_watches = average_error :=
sorry

end NUMINAMATH_CALUDE_average_timing_error_l663_66373


namespace NUMINAMATH_CALUDE_arielle_age_l663_66358

theorem arielle_age (elvie_age arielle_age : ℕ) : 
  elvie_age = 10 → 
  elvie_age + arielle_age + elvie_age * arielle_age = 131 → 
  arielle_age = 11 := by
sorry

end NUMINAMATH_CALUDE_arielle_age_l663_66358


namespace NUMINAMATH_CALUDE_rosa_pages_last_week_l663_66331

-- Define the total number of pages called
def total_pages : ℝ := 18.8

-- Define the number of pages called this week
def pages_this_week : ℝ := 8.6

-- Define the number of pages called last week
def pages_last_week : ℝ := total_pages - pages_this_week

-- Theorem to prove
theorem rosa_pages_last_week : pages_last_week = 10.2 := by
  sorry

end NUMINAMATH_CALUDE_rosa_pages_last_week_l663_66331


namespace NUMINAMATH_CALUDE_probability_equals_three_fourths_l663_66366

/-- The set S in R^2 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | -2 ≤ p.2 ∧ p.2 ≤ |p.1| ∧ -2 ≤ p.1 ∧ p.1 ≤ 2}

/-- The subset of S where |x| + |y| < 2 -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ S ∧ |p.1| + |p.2| < 2}

/-- The area of a set in R^2 -/
noncomputable def area (A : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem -/
theorem probability_equals_three_fourths :
  area T / area S = 3/4 := by sorry

end NUMINAMATH_CALUDE_probability_equals_three_fourths_l663_66366


namespace NUMINAMATH_CALUDE_gcd_78_143_l663_66310

theorem gcd_78_143 : Nat.gcd 78 143 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_78_143_l663_66310


namespace NUMINAMATH_CALUDE_inequality_transformation_l663_66355

theorem inequality_transformation (a b : ℝ) (h : a > b) : -3 * a < -3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l663_66355


namespace NUMINAMATH_CALUDE_f_increasing_and_range_l663_66304

-- Define the function f and its properties
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x > 0 → f x > 0) ∧
  (f (-1) = -2)

-- Theorem statement
theorem f_increasing_and_range (f : ℝ → ℝ) (hf : f_properties f) :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (Set.range (fun x => f x) ∩ Set.Icc (-2) 1 = Set.Icc (-4) 2) :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_and_range_l663_66304


namespace NUMINAMATH_CALUDE_only_physical_education_survey_census_suitable_physical_education_survey_census_suitable_light_bulb_survey_not_census_suitable_tv_program_survey_not_census_suitable_national_height_survey_not_census_suitable_l663_66397

/-- Represents a survey --/
inductive Survey
  | PhysicalEducationScores
  | LightBulbLifespan
  | TVProgramPreferences
  | NationalStudentHeight

/-- Checks if a survey satisfies the conditions for a census --/
def isCensusSuitable (s : Survey) : Prop :=
  match s with
  | Survey.PhysicalEducationScores => true
  | _ => false

/-- Theorem stating that only the physical education scores survey is suitable for a census --/
theorem only_physical_education_survey_census_suitable :
  ∀ s : Survey, isCensusSuitable s ↔ s = Survey.PhysicalEducationScores :=
by sorry

/-- Proof that the physical education scores survey is suitable for a census --/
theorem physical_education_survey_census_suitable :
  isCensusSuitable Survey.PhysicalEducationScores :=
by sorry

/-- Proof that the light bulb lifespan survey is not suitable for a census --/
theorem light_bulb_survey_not_census_suitable :
  ¬ isCensusSuitable Survey.LightBulbLifespan :=
by sorry

/-- Proof that the TV program preferences survey is not suitable for a census --/
theorem tv_program_survey_not_census_suitable :
  ¬ isCensusSuitable Survey.TVProgramPreferences :=
by sorry

/-- Proof that the national student height survey is not suitable for a census --/
theorem national_height_survey_not_census_suitable :
  ¬ isCensusSuitable Survey.NationalStudentHeight :=
by sorry

end NUMINAMATH_CALUDE_only_physical_education_survey_census_suitable_physical_education_survey_census_suitable_light_bulb_survey_not_census_suitable_tv_program_survey_not_census_suitable_national_height_survey_not_census_suitable_l663_66397


namespace NUMINAMATH_CALUDE_abs_sum_inequality_solution_set_square_sum_geq_sqrt_product_sum_l663_66341

-- Part Ⅰ
theorem abs_sum_inequality_solution_set (x : ℝ) :
  (|2 + x| + |2 - x| ≤ 4) ↔ (-2 ≤ x ∧ x ≤ 2) := by sorry

-- Part Ⅱ
theorem square_sum_geq_sqrt_product_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  a^2 + b^2 ≥ Real.sqrt (a * b) * (a + b) := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_solution_set_square_sum_geq_sqrt_product_sum_l663_66341


namespace NUMINAMATH_CALUDE_complex_equation_solution_l663_66342

theorem complex_equation_solution (a b : ℝ) (h : (3 + 4*I) * (1 + a*I) = b*I) : a = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l663_66342


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_equation_l663_66376

theorem positive_integer_solutions_of_equation : 
  {(x, y) : ℕ × ℕ | x + y + x * y = 2008} = 
  {(6, 286), (286, 6), (40, 48), (48, 40)} := by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_equation_l663_66376


namespace NUMINAMATH_CALUDE_final_sum_is_130_l663_66349

/-- Represents the financial state of Earl, Fred, and Greg --/
structure FinancialState where
  earl : Int
  fred : Int
  greg : Int

/-- Represents the debts between Earl, Fred, and Greg --/
structure Debts where
  earl_to_fred : Int
  fred_to_greg : Int
  greg_to_earl : Int

/-- Calculates the final amounts for Earl and Greg after settling all debts --/
def settle_debts (initial : FinancialState) (debts : Debts) : Int × Int :=
  let earl_final := initial.earl - debts.earl_to_fred + debts.greg_to_earl
  let greg_final := initial.greg + debts.fred_to_greg - debts.greg_to_earl
  (earl_final, greg_final)

/-- Theorem stating that Greg and Earl will have $130 together after settling all debts --/
theorem final_sum_is_130 (initial : FinancialState) (debts : Debts) :
  initial.earl = 90 →
  initial.fred = 48 →
  initial.greg = 36 →
  debts.earl_to_fred = 28 →
  debts.fred_to_greg = 32 →
  debts.greg_to_earl = 40 →
  let (earl_final, greg_final) := settle_debts initial debts
  earl_final + greg_final = 130 := by
  sorry

#check final_sum_is_130

end NUMINAMATH_CALUDE_final_sum_is_130_l663_66349
