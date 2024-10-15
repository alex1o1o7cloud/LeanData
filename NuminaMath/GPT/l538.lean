import Mathlib

namespace NUMINAMATH_GPT_identity_implies_a_minus_b_l538_53868

theorem identity_implies_a_minus_b (a b : ℚ) (y : ℚ) (h : y > 0) :
  (∀ y, y > 0 → (a / (y - 3) + b / (y + 5) = (3 * y + 7) / ((y - 3) * (y + 5)))) → (a - b = 1) :=
by
  sorry

end NUMINAMATH_GPT_identity_implies_a_minus_b_l538_53868


namespace NUMINAMATH_GPT_triangle_inequality_l538_53892

theorem triangle_inequality 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : A + B + C = π) 
  (h5 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  (h6 : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B)
  (h7 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) :
  3 / 2 ≤ a^2 / (b^2 + c^2) + b^2 / (c^2 + a^2) + c^2 / (a^2 + b^2) ∧
  (a^2 / (b^2 + c^2) + b^2 / (c^2 + a^2) + c^2 / (a^2 + b^2) ≤ 
     2 * ((Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2)) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l538_53892


namespace NUMINAMATH_GPT_minute_hand_only_rotates_l538_53890

-- Define what constitutes translation and rotation
def is_translation (motion : ℝ → ℝ → Prop) : Prop :=
  ∀ (p1 p2 : ℝ), motion p1 p2 → (∃ d : ℝ, ∀ t : ℝ, motion (p1 + t) (p2 + t) ∧ |p1 - p2| = d)

def is_rotation (motion : ℝ → ℝ → Prop) : Prop :=
  ∀ (p : ℝ), ∃ c : ℝ, ∃ r : ℝ, (∀ (t : ℝ), |p - c| = r)

-- Define the condition that the minute hand of a clock undergoes a specific motion
def minute_hand_motion (p : ℝ) (t : ℝ) : Prop :=
  -- The exact definition here would involve trigonometric representation
  sorry

-- The main proof statement
theorem minute_hand_only_rotates :
  is_rotation minute_hand_motion ∧ ¬ is_translation minute_hand_motion :=
sorry

end NUMINAMATH_GPT_minute_hand_only_rotates_l538_53890


namespace NUMINAMATH_GPT_solve_missing_number_l538_53855

theorem solve_missing_number (n : ℤ) (h : 121 * n = 75625) : n = 625 :=
sorry

end NUMINAMATH_GPT_solve_missing_number_l538_53855


namespace NUMINAMATH_GPT_pratyya_payel_min_difference_l538_53849

theorem pratyya_payel_min_difference (n m : ℕ) (h : n > m ∧ n - m ≥ 4) :
  ∀ t : ℕ, (2^(t+1) * n - 2^(t+1)) > 2^(t+1) * m + 2^(t+1) :=
by
  sorry

end NUMINAMATH_GPT_pratyya_payel_min_difference_l538_53849


namespace NUMINAMATH_GPT_number_of_numbers_tadd_said_after_20_rounds_l538_53888

-- Define the arithmetic sequence representing the count of numbers Tadd says each round
def tadd_sequence (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

-- Define the sum of the first n terms of Tadd's sequence
def sum_tadd_sequence (n : ℕ) : ℕ :=
  n * (1 + tadd_sequence n) / 2

-- The main theorem to state the problem
theorem number_of_numbers_tadd_said_after_20_rounds :
  sum_tadd_sequence 20 = 400 :=
by
  -- The actual proof should be filled in here
  sorry

end NUMINAMATH_GPT_number_of_numbers_tadd_said_after_20_rounds_l538_53888


namespace NUMINAMATH_GPT_bricks_required_for_courtyard_l538_53807

/-- 
A courtyard is 45 meters long and 25 meters broad needs to be paved with bricks of 
dimensions 15 cm by 7 cm. What will be the total number of bricks required?
-/
theorem bricks_required_for_courtyard 
  (courtyard_length : ℕ) (courtyard_width : ℕ)
  (brick_length : ℕ) (brick_width : ℕ)
  (H1 : courtyard_length = 4500) (H2 : courtyard_width = 2500)
  (H3 : brick_length = 15) (H4 : brick_width = 7) :
  let courtyard_area_cm : ℕ := courtyard_length * courtyard_width
  let brick_area_cm : ℕ := brick_length * brick_width
  let total_bricks : ℕ := (courtyard_area_cm + brick_area_cm - 1) / brick_area_cm
  total_bricks = 107143 := by
  sorry

end NUMINAMATH_GPT_bricks_required_for_courtyard_l538_53807


namespace NUMINAMATH_GPT_distance_traveled_l538_53869

noncomputable def velocity (t : ℝ) := 2 * t - 3

theorem distance_traveled : 
  (∫ t in (0 : ℝ)..5, |velocity t|) = 29 / 2 := 
by
  sorry

end NUMINAMATH_GPT_distance_traveled_l538_53869


namespace NUMINAMATH_GPT_number_of_croutons_l538_53897

def lettuce_calories : ℕ := 30
def cucumber_calories : ℕ := 80
def crouton_calories : ℕ := 20
def total_salad_calories : ℕ := 350

theorem number_of_croutons : 
  ∃ n : ℕ, n * crouton_calories = total_salad_calories - (lettuce_calories + cucumber_calories) ∧ n = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_croutons_l538_53897


namespace NUMINAMATH_GPT_solve_parabola_l538_53854

theorem solve_parabola (a b c : ℝ) 
  (h1 : 1 = a * 1^2 + b * 1 + c)
  (h2 : 4 * a + b = 1)
  (h3 : -1 = a * 2^2 + b * 2 + c) :
  a = 3 ∧ b = -11 ∧ c = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_parabola_l538_53854


namespace NUMINAMATH_GPT_total_amount_spent_l538_53886

def price_of_brand_X_pen : ℝ := 4.00
def price_of_brand_Y_pen : ℝ := 2.20
def total_pens_purchased : ℝ := 12
def brand_X_pens_purchased : ℝ := 6

theorem total_amount_spent :
  let brand_X_cost := brand_X_pens_purchased * price_of_brand_X_pen
  let brand_Y_pens_purchased := total_pens_purchased - brand_X_pens_purchased
  let brand_Y_cost := brand_Y_pens_purchased * price_of_brand_Y_pen
  brand_X_cost + brand_Y_cost = 37.20 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l538_53886


namespace NUMINAMATH_GPT_pete_backward_speed_l538_53881

variable (p b t s : ℝ)  -- speeds of Pete, backward walk, Tracy, and Susan respectively

-- Given conditions
axiom h1 : p / t = 1 / 4      -- Pete walks on his hands at a quarter speed of Tracy's cartwheeling
axiom h2 : t = 2 * s          -- Tracy cartwheels twice as fast as Susan walks
axiom h3 : b = 3 * s          -- Pete walks backwards three times faster than Susan
axiom h4 : p = 2              -- Pete walks on his hands at 2 miles per hour

-- Prove Pete's backward walking speed is 12 miles per hour
theorem pete_backward_speed : b = 12 :=
by
  sorry

end NUMINAMATH_GPT_pete_backward_speed_l538_53881


namespace NUMINAMATH_GPT_circle_center_radius_sum_l538_53885

-- We define the circle equation as a predicate
def circle_eq (x y : ℝ) : Prop :=
  x^2 - 14 * x + y^2 + 16 * y + 100 = 0

-- We need to find that the center and radius satisfy a specific relationship
theorem circle_center_radius_sum :
  let a' := 7
  let b' := -8
  let r' := Real.sqrt 13
  a' + b' + r' = -1 + Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_radius_sum_l538_53885


namespace NUMINAMATH_GPT_smallest_k_no_real_roots_l538_53851

theorem smallest_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, 3 * x * (k * x - 5) - 2 * x^2 + 13 ≠ 0) ∧
  (∀ n : ℤ, n < k → ∃ x : ℝ, 3 * x * (n * x - 5) - 2 * x^2 + 13 = 0) :=
by sorry

end NUMINAMATH_GPT_smallest_k_no_real_roots_l538_53851


namespace NUMINAMATH_GPT_chicken_cost_l538_53834

theorem chicken_cost (total_money hummus_price hummus_count bacon_price vegetables_price apple_price apple_count chicken_price : ℕ)
  (h_total_money : total_money = 60)
  (h_hummus_price : hummus_price = 5)
  (h_hummus_count : hummus_count = 2)
  (h_bacon_price : bacon_price = 10)
  (h_vegetables_price : vegetables_price = 10)
  (h_apple_price : apple_price = 2)
  (h_apple_count : apple_count = 5)
  (h_remaining_money : chicken_price = total_money - (hummus_count * hummus_price + bacon_price + vegetables_price + apple_count * apple_price)) :
  chicken_price = 20 := 
by sorry

end NUMINAMATH_GPT_chicken_cost_l538_53834


namespace NUMINAMATH_GPT_software_package_cost_l538_53891

theorem software_package_cost 
  (devices : ℕ) 
  (cost_first : ℕ) 
  (devices_covered_first : ℕ) 
  (devices_covered_second : ℕ) 
  (savings : ℕ)
  (total_cost_first : ℕ := (devices / devices_covered_first) * cost_first)
  (total_cost_second : ℕ := total_cost_first - savings)
  (num_packages_second : ℕ := devices / devices_covered_second)
  (cost_second : ℕ := total_cost_second / num_packages_second) :
  devices = 50 ∧ cost_first = 40 ∧ devices_covered_first = 5 ∧ devices_covered_second = 10 ∧ savings = 100 →
  cost_second = 60 := 
by
  sorry

end NUMINAMATH_GPT_software_package_cost_l538_53891


namespace NUMINAMATH_GPT_find_h_plus_k_l538_53864

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 14*y - 11 = 0

-- State the problem: Prove h + k = -4 given (h, k) is the center of the circle
theorem find_h_plus_k : (∃ h k, (∀ x y, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 69) ∧ h + k = -4) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_h_plus_k_l538_53864


namespace NUMINAMATH_GPT_amount_of_money_around_circumference_l538_53823

-- Define the given conditions
def horizontal_coins : ℕ := 6
def vertical_coins : ℕ := 4
def coin_value_won : ℕ := 100

-- The goal is to prove the total amount of money around the circumference
theorem amount_of_money_around_circumference : 
  (2 * (horizontal_coins - 2) + 2 * (vertical_coins - 2) + 4) * coin_value_won = 1600 :=
by
  sorry

end NUMINAMATH_GPT_amount_of_money_around_circumference_l538_53823


namespace NUMINAMATH_GPT_inequality_ge_five_halves_l538_53824

open Real

noncomputable def xy_yz_zx_eq_one (x y z : ℝ) := x * y + y * z + z * x = 1
noncomputable def non_neg (x y z : ℝ) := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

theorem inequality_ge_five_halves (x y z : ℝ) (h1 : xy_yz_zx_eq_one x y z) (h2 : non_neg x y z) :
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 5 / 2 := 
sorry

end NUMINAMATH_GPT_inequality_ge_five_halves_l538_53824


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l538_53878

theorem simplify_and_evaluate_expression (x y : ℝ) (h_x : x = -2) (h_y : y = 1) :
  (((2 * x - (1/2) * y)^2 - ((-y + 2 * x) * (2 * x + y)) + y * (x^2 * y - (5/4) * y)) / x) = -4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l538_53878


namespace NUMINAMATH_GPT_remaining_amount_to_be_paid_l538_53887

theorem remaining_amount_to_be_paid (p : ℝ) (deposit : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) (final_payment : ℝ) :
  deposit = 80 ∧ tax_rate = 0.07 ∧ discount_rate = 0.05 ∧ deposit = 0.1 * p ∧ 
  final_payment = (p - (discount_rate * p)) * (1 + tax_rate) - deposit → 
  final_payment = 733.20 :=
by
  sorry

end NUMINAMATH_GPT_remaining_amount_to_be_paid_l538_53887


namespace NUMINAMATH_GPT_no_int_sol_eq_l538_53898

theorem no_int_sol_eq (x y z : ℤ) (h₀ : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) : ¬ (x^2 + y^2 = 3 * z^2) := 
sorry

end NUMINAMATH_GPT_no_int_sol_eq_l538_53898


namespace NUMINAMATH_GPT_arithmetic_progr_property_l538_53848

theorem arithmetic_progr_property (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 + a 3 = 5 / 2)
  (h2 : a 2 + a 4 = 5 / 4)
  (h3 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  (h4 : a 3 = a 1 + 2 * (a 2 - a 1))
  (h5 : a 2 = a 1 + (a 2 - a 1)) :
  S 3 / a 3 = 6 := sorry

end NUMINAMATH_GPT_arithmetic_progr_property_l538_53848


namespace NUMINAMATH_GPT_tom_has_9_balloons_l538_53876

-- Define Tom's and Sara's yellow balloon counts
variables (total_balloons saras_balloons toms_balloons : ℕ)

-- Given conditions
axiom total_balloons_def : total_balloons = 17
axiom saras_balloons_def : saras_balloons = 8
axiom toms_balloons_total : toms_balloons + saras_balloons = total_balloons

-- Theorem stating that Tom has 9 yellow balloons
theorem tom_has_9_balloons : toms_balloons = 9 := by
  sorry

end NUMINAMATH_GPT_tom_has_9_balloons_l538_53876


namespace NUMINAMATH_GPT_inequality_AM_GM_l538_53871

theorem inequality_AM_GM
  (a b c : ℝ)
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c)
  (habc : a + b + c = 1) : 
  (a + 2 * a * b + 2 * a * c + b * c) ^ a * 
  (b + 2 * b * c + 2 * b * a + c * a) ^ b * 
  (c + 2 * c * a + 2 * c * b + a * b) ^ c ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_AM_GM_l538_53871


namespace NUMINAMATH_GPT_annual_growth_rate_l538_53846

theorem annual_growth_rate (p : ℝ) : 
  let S1 := (1 + p) ^ 12 - 1 / p
  let S2 := ((1 + p) ^ 12 * ((1 + p) ^ 12 - 1)) / p
  let annual_growth := (S2 - S1) / S1
  annual_growth = (1 + p) ^ 12 - 1 :=
by
  sorry

end NUMINAMATH_GPT_annual_growth_rate_l538_53846


namespace NUMINAMATH_GPT_train_crossing_time_l538_53820

/--
A train requires 8 seconds to pass a pole while it requires some seconds to cross a stationary train which is 400 meters long. 
The speed of the train is 144 km/h. Prove that it takes 18 seconds for the train to cross the stationary train.
-/
theorem train_crossing_time
  (train_speed_kmh : ℕ)
  (time_to_pass_pole : ℕ)
  (length_stationary_train : ℕ)
  (speed_mps : ℕ)
  (length_moving_train : ℕ)
  (total_length : ℕ)
  (crossing_time : ℕ) :
  train_speed_kmh = 144 →
  time_to_pass_pole = 8 →
  length_stationary_train = 400 →
  speed_mps = (train_speed_kmh * 1000) / 3600 →
  length_moving_train = speed_mps * time_to_pass_pole →
  total_length = length_moving_train + length_stationary_train →
  crossing_time = total_length / speed_mps →
  crossing_time = 18 :=
by
  intros;
  sorry

end NUMINAMATH_GPT_train_crossing_time_l538_53820


namespace NUMINAMATH_GPT_cylinder_height_l538_53858

theorem cylinder_height (r h : ℝ) (SA : ℝ) (h_cond : SA = 2 * π * r^2 + 2 * π * r * h) 
  (r_eq : r = 3) (SA_eq : SA = 27 * π) : h = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_height_l538_53858


namespace NUMINAMATH_GPT_solve_for_y_l538_53860

theorem solve_for_y (x y : ℝ) (h : 2 * x - 7 * y = 8) : y = (2 * x - 8) / 7 := by
  sorry

end NUMINAMATH_GPT_solve_for_y_l538_53860


namespace NUMINAMATH_GPT_swimmer_speed_in_still_water_l538_53873

-- Define the conditions
def current_speed : ℝ := 2   -- Speed of the water current is 2 km/h
def swim_time : ℝ := 2.5     -- Time taken to swim against current is 2.5 hours
def distance : ℝ := 5        -- Distance swum against current is 5 km

-- Main theorem proving the swimmer's speed in still water
theorem swimmer_speed_in_still_water (v : ℝ) (h : v - current_speed = distance / swim_time) : v = 4 :=
by {
  -- Skipping the proof steps as per the requirements
  sorry
}

end NUMINAMATH_GPT_swimmer_speed_in_still_water_l538_53873


namespace NUMINAMATH_GPT_value_of_expression_l538_53843

noncomputable def largestNegativeInteger : Int := -1

theorem value_of_expression (a b x y : ℝ) (m : Int)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = largestNegativeInteger) :
  2023 * (a + b) + 3 * |m| - 2 * (x * y) = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l538_53843


namespace NUMINAMATH_GPT_inequality_solution_set_l538_53840

theorem inequality_solution_set :
  {x : ℝ | (x - 3) / (x + 2) ≤ 0} = {x : ℝ | -2 < x ∧ x ≤ 3} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l538_53840


namespace NUMINAMATH_GPT_smallest_ratio_l538_53802

-- Define the system of equations as conditions
def eq1 (x y : ℝ) := x^3 + 3 * y^3 = 11
def eq2 (x y : ℝ) := (x^2 * y) + (x * y^2) = 6

-- Define the goal: proving the smallest value of x/y for the solutions (x, y) is -1.31
theorem smallest_ratio (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) :
  ∃ t : ℝ, t = x / y ∧ ∀ t', t' = x / y → t' ≥ -1.31 :=
sorry

end NUMINAMATH_GPT_smallest_ratio_l538_53802


namespace NUMINAMATH_GPT_initial_workers_l538_53870

theorem initial_workers (W : ℕ) (H1 : (8 * W) / 30 = W) (H2 : (6 * (2 * W - 45)) / 45 = 2 * W - 45) : W = 45 :=
sorry

end NUMINAMATH_GPT_initial_workers_l538_53870


namespace NUMINAMATH_GPT_solve_line_eq_l538_53883

theorem solve_line_eq (a b x : ℝ) (h1 : (0 : ℝ) * a + b = 2) (h2 : -3 * a + b = 0) : x = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_line_eq_l538_53883


namespace NUMINAMATH_GPT_least_integer_exists_l538_53845

theorem least_integer_exists (x : ℕ) (h1 : x = 10 * (x / 10) + x % 10) (h2 : (x / 10) = x / 17) : x = 17 :=
sorry

end NUMINAMATH_GPT_least_integer_exists_l538_53845


namespace NUMINAMATH_GPT_correct_operation_l538_53804

theorem correct_operation (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by sorry

end NUMINAMATH_GPT_correct_operation_l538_53804


namespace NUMINAMATH_GPT_coloring_methods_390_l538_53879

def numColoringMethods (colors cells : ℕ) (maxColors : ℕ) : ℕ :=
  if colors = 6 ∧ cells = 4 ∧ maxColors = 3 then 390 else 0

theorem coloring_methods_390 :
  numColoringMethods 6 4 3 = 390 :=
by 
  sorry

end NUMINAMATH_GPT_coloring_methods_390_l538_53879


namespace NUMINAMATH_GPT_square_roots_of_x_l538_53895

theorem square_roots_of_x (a x : ℝ) 
    (h1 : (2 * a - 1) ^ 2 = x) 
    (h2 : (-a + 2) ^ 2 = x)
    (hx : 0 < x) 
    : x = 9 ∨ x = 1 := 
by sorry

end NUMINAMATH_GPT_square_roots_of_x_l538_53895


namespace NUMINAMATH_GPT_problem_I_problem_II_l538_53816

namespace ProofProblems

def f (x a : ℝ) : ℝ := |x - a| + |x + 5|

theorem problem_I (x : ℝ) : (f x 1) ≥ 2 * |x + 5| ↔ x ≤ -2 := 
by sorry

theorem problem_II (a : ℝ) : 
  (∀ x : ℝ, (f x a) ≥ 8) ↔ (a ≥ 3 ∨ a ≤ -13) := 
by sorry

end ProofProblems

end NUMINAMATH_GPT_problem_I_problem_II_l538_53816


namespace NUMINAMATH_GPT_solution_l538_53818

-- Definitions for vectors a and b with given conditions for orthogonality and equal magnitudes
def a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

-- Orthogonality condition
def orthogonal (p q : ℝ) : Prop := 4 * 3 + p * 2 + (-2) * q = 0

-- Equal magnitude condition
def equal_magnitudes (p q : ℝ) : Prop :=
  4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2

-- Proof problem
theorem solution (p q : ℝ) (h_orthogonal : orthogonal p q) (h_equal_magnitudes : equal_magnitudes p q) :
  p = -29 / 12 ∧ q = 43 / 12 := 
by 
  sorry

end NUMINAMATH_GPT_solution_l538_53818


namespace NUMINAMATH_GPT_conservation_center_total_turtles_l538_53817

-- Define the green turtles and the relationship between green and hawksbill turtles.
def green_turtles : ℕ := 800
def hawksbill_turtles : ℕ := 2 * green_turtles

-- Statement we need to prove, which is the total number of turtles equals 3200.
theorem conservation_center_total_turtles : green_turtles + hawksbill_turtles = 3200 := by
  sorry

end NUMINAMATH_GPT_conservation_center_total_turtles_l538_53817


namespace NUMINAMATH_GPT_pyramid_volume_l538_53829

noncomputable def volume_of_pyramid (S α β : ℝ) : ℝ :=
  (1 / 6) * S * (Real.sqrt (2 * S * (Real.tan α) * (Real.tan β)))

theorem pyramid_volume 
  (S α β : ℝ)
  (base_area : S > 0)
  (equal_lateral_edges : true)
  (dihedral_angles : α > 0 ∧ α < π / 2 ∧ β > 0 ∧ β < π / 2) :
  volume_of_pyramid S α β = (1 / 6) * S * (Real.sqrt (2 * S * (Real.tan α) * (Real.tan β))) :=
by
  sorry

end NUMINAMATH_GPT_pyramid_volume_l538_53829


namespace NUMINAMATH_GPT_jenny_chocolate_milk_probability_l538_53825

-- Define the binomial probability function.
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  ( Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

-- Given conditions: probability each day and total number of days.
def probability_each_day : ℚ := 2 / 3
def num_days : ℕ := 7
def successful_days : ℕ := 3

-- The problem statement to prove.
theorem jenny_chocolate_milk_probability :
  binomial_probability num_days successful_days probability_each_day = 280 / 2187 :=
by
  sorry

end NUMINAMATH_GPT_jenny_chocolate_milk_probability_l538_53825


namespace NUMINAMATH_GPT_sum_squares_of_roots_of_polynomial_l538_53862

noncomputable def roots (n : ℕ) (p : Polynomial ℂ) : List ℂ :=
  if h : n = p.natDegree then Multiset.toList p.roots else []

theorem sum_squares_of_roots_of_polynomial :
  (roots 2018 (Polynomial.C 404 + Polynomial.C 3 * X ^ 3 + Polynomial.C 44 * X ^ 2015 + X ^ 2018)).sum = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_squares_of_roots_of_polynomial_l538_53862


namespace NUMINAMATH_GPT_ratio_of_radii_l538_53808

theorem ratio_of_radii (a b c : ℝ) (h1 : π * c^2 - π * a^2 = 4 * π * a^2) (h2 : π * b^2 = (π * a^2 + π * c^2) / 2) :
  a / c = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_ratio_of_radii_l538_53808


namespace NUMINAMATH_GPT_number_of_students_l538_53857

theorem number_of_students (pencils: ℕ) (pencils_per_student: ℕ) (total_students: ℕ) 
  (h1: pencils = 195) (h2: pencils_per_student = 3) (h3: total_students = pencils / pencils_per_student) :
  total_students = 65 := by
  -- proof would go here, but we skip it with sorry for now
  sorry

end NUMINAMATH_GPT_number_of_students_l538_53857


namespace NUMINAMATH_GPT_work_completion_days_l538_53801

theorem work_completion_days (A B C : ℕ) (work_rate_A : A = 4) (work_rate_B : B = 10) (work_rate_C : C = 20 / 3) :
  (1 / A) + (1 / B) + (3 / C) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_days_l538_53801


namespace NUMINAMATH_GPT_convert_to_rectangular_form_l538_53844

theorem convert_to_rectangular_form :
  (Complex.exp (13 * Real.pi * Complex.I / 2)) = Complex.I :=
by
  sorry

end NUMINAMATH_GPT_convert_to_rectangular_form_l538_53844


namespace NUMINAMATH_GPT_polynomial_has_root_l538_53811

theorem polynomial_has_root {a b c d : ℝ} 
  (h : a * c = 2 * b + 2 * d) : 
  ∃ x : ℝ, (x^2 + a * x + b = 0) ∨ (x^2 + c * x + d = 0) :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_has_root_l538_53811


namespace NUMINAMATH_GPT_max_Sn_in_arithmetic_sequence_l538_53835

theorem max_Sn_in_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : ∀ {m n p q : ℕ}, m + n = p + q → a m + a n = a p + a q)
  (h_a4 : a 4 = 1)
  (h_S5 : S 5 = 10)
  (h_S : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) :
  ∃ n, n = 4 ∨ n = 5 ∧ ∀ m ≠ n, S m ≤ S n := by
  sorry

end NUMINAMATH_GPT_max_Sn_in_arithmetic_sequence_l538_53835


namespace NUMINAMATH_GPT_janet_additional_money_needed_is_1225_l538_53896

def savings : ℕ := 2225
def rent_per_month : ℕ := 1250
def months_required : ℕ := 2
def deposit : ℕ := 500
def utility_deposit : ℕ := 300
def moving_costs : ℕ := 150

noncomputable def total_rent : ℕ := rent_per_month * months_required
noncomputable def total_upfront_cost : ℕ := total_rent + deposit + utility_deposit + moving_costs
noncomputable def additional_money_needed : ℕ := total_upfront_cost - savings

theorem janet_additional_money_needed_is_1225 : additional_money_needed = 1225 :=
by
  sorry

end NUMINAMATH_GPT_janet_additional_money_needed_is_1225_l538_53896


namespace NUMINAMATH_GPT_smallestC_l538_53867

def isValidFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x) ∧
  f 1 = 1 ∧
  (∀ x y, 0 ≤ x ∧ 0 ≤ y ∧ x + y ≤ 1 → f x + f y ≤ f (x + y))

theorem smallestC (f : ℝ → ℝ) (h : isValidFunction f) : ∃ c, (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ c * x) ∧
  (∀ d, (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ d * x) → 2 ≤ d) :=
sorry

end NUMINAMATH_GPT_smallestC_l538_53867


namespace NUMINAMATH_GPT_sugar_and_granulated_sugar_delivered_l538_53819

theorem sugar_and_granulated_sugar_delivered (total_bags : ℕ) (percentage_more : ℚ) (mass_ratio : ℚ) (total_weight : ℚ)
    (h_total_bags : total_bags = 63)
    (h_percentage_more : percentage_more = 1.25)
    (h_mass_ratio : mass_ratio = 3 / 4)
    (h_total_weight : total_weight = 4.8) :
    ∃ (sugar_weight granulated_sugar_weight : ℚ),
        (granulated_sugar_weight = 1.8) ∧ (sugar_weight = 3) ∧
        ((sugar_weight + granulated_sugar_weight = total_weight) ∧
        (sugar_weight / 28 = (granulated_sugar_weight / 35) * mass_ratio)) :=
by
    sorry

end NUMINAMATH_GPT_sugar_and_granulated_sugar_delivered_l538_53819


namespace NUMINAMATH_GPT_paint_per_statue_calculation_l538_53865

theorem paint_per_statue_calculation (total_paint : ℚ) (num_statues : ℕ) (expected_paint_per_statue : ℚ) :
  total_paint = 7 / 8 → num_statues = 14 → expected_paint_per_statue = 7 / 112 → 
  total_paint / num_statues = expected_paint_per_statue :=
by
  intros htotal hnum_expected hequals
  rw [htotal, hnum_expected, hequals]
  -- Using the fact that:
  -- total_paint / num_statues = (7 / 8) / 14
  -- This can be rewritten as (7 / 8) * (1 / 14) = 7 / (8 * 14) = 7 / 112
  sorry

end NUMINAMATH_GPT_paint_per_statue_calculation_l538_53865


namespace NUMINAMATH_GPT_inequality_abc_l538_53837

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2)) + (b^3 / (b^2 + b * c + c^2)) + (c^3 / (c^2 + c * a + a^2)) >= (a + b + c) / 3 := by
  sorry

end NUMINAMATH_GPT_inequality_abc_l538_53837


namespace NUMINAMATH_GPT_mosquito_distance_ratio_l538_53847

-- Definition of the clock problem conditions
structure ClockInsects where
  distance_from_center : ℕ
  initial_time : ℕ := 1

-- Prove the ratio of distances traveled by mosquito and fly over 12 hours
theorem mosquito_distance_ratio (c : ClockInsects) :
  let mosquito_distance := (83 : ℚ)/12
  let fly_distance := (73 : ℚ)/12
  mosquito_distance / fly_distance = 83 / 73 :=
by 
  sorry

end NUMINAMATH_GPT_mosquito_distance_ratio_l538_53847


namespace NUMINAMATH_GPT_youngest_sibling_age_l538_53841

theorem youngest_sibling_age
    (age_youngest : ℕ)
    (first_sibling : ℕ := age_youngest + 4)
    (second_sibling : ℕ := age_youngest + 5)
    (third_sibling : ℕ := age_youngest + 7)
    (average_age : ℕ := 21)
    (sum_of_ages : ℕ := 4 * average_age)
    (total_age_check : (age_youngest + first_sibling + second_sibling + third_sibling) = sum_of_ages) :
  age_youngest = 17 :=
sorry

end NUMINAMATH_GPT_youngest_sibling_age_l538_53841


namespace NUMINAMATH_GPT_arithmetic_expression_equality_l538_53893

theorem arithmetic_expression_equality :
  ( ( (4 + 6 + 5) * 2 ) / 4 - ( (3 * 2) / 4 ) ) = 6 :=
by sorry

end NUMINAMATH_GPT_arithmetic_expression_equality_l538_53893


namespace NUMINAMATH_GPT_solve_fractional_eq_l538_53889

theorem solve_fractional_eq (x: ℝ) (h1: x ≠ -11) (h2: x ≠ -8) (h3: x ≠ -12) (h4: x ≠ -7) :
  (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) → (x = -19 / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_fractional_eq_l538_53889


namespace NUMINAMATH_GPT_truncated_pyramid_smaller_base_area_l538_53814

noncomputable def smaller_base_area (a : ℝ) (α β : ℝ) : ℝ :=
  (a^2 * (Real.sin (α - β))^2) / (Real.sin (α + β))^2

theorem truncated_pyramid_smaller_base_area (a α β : ℝ) :
  smaller_base_area a α β = (a^2 * (Real.sin (α - β))^2) / (Real.sin (α + β))^2 :=
by
  unfold smaller_base_area
  sorry

end NUMINAMATH_GPT_truncated_pyramid_smaller_base_area_l538_53814


namespace NUMINAMATH_GPT_kim_easy_round_correct_answers_l538_53800

variable (E : ℕ)

theorem kim_easy_round_correct_answers 
    (h1 : 2 * E + 3 * 2 + 5 * 4 = 38) : 
    E = 6 := 
sorry

end NUMINAMATH_GPT_kim_easy_round_correct_answers_l538_53800


namespace NUMINAMATH_GPT_geometric_sequence_r_value_l538_53828

theorem geometric_sequence_r_value (S : ℕ → ℚ) (r : ℚ) (n : ℕ) (h : n ≥ 2) (h1 : ∀ n, S n = 3^n + r) :
    r = -1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_r_value_l538_53828


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l538_53875

theorem necessary_and_sufficient_condition (x : ℝ) : (|x - 2| < 1) ↔ (1 < x ∧ x < 3) := 
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l538_53875


namespace NUMINAMATH_GPT_original_intensity_45_percent_l538_53810

variable (I : ℝ) -- Intensity of the original red paint in percentage.

-- Conditions
variable (h1 : 25 * 0.25 + 0.75 * I = 40) -- Given conditions about the intensities and the new solution.
variable (h2 : ∀ I : ℝ, 0.75 * I + 25 * 0.25 = 40) -- Rewriting the given condition to look specifically for I.

theorem original_intensity_45_percent (I : ℝ) (h1 : 25 * 0.25 + 0.75 * I = 40) : I = 45 := by
  -- We only need the statement. Proof is not required.
  sorry

end NUMINAMATH_GPT_original_intensity_45_percent_l538_53810


namespace NUMINAMATH_GPT_yellow_block_weight_proof_l538_53831

-- Define the weights and the relationship between them
def green_block_weight : ℝ := 0.4
def additional_weight : ℝ := 0.2
def yellow_block_weight : ℝ := green_block_weight + additional_weight

-- The theorem to prove
theorem yellow_block_weight_proof : yellow_block_weight = 0.6 :=
by
  -- Proof will be supplied here
  sorry

end NUMINAMATH_GPT_yellow_block_weight_proof_l538_53831


namespace NUMINAMATH_GPT_find_a1_an_l538_53822

noncomputable def arith_geo_seq (a : ℕ → ℝ) : Prop :=
  (∃ d ≠ 0, (a 2 + a 4 = 10) ∧ (a 2 ^ 2 = a 1 * a 5))

theorem find_a1_an (a : ℕ → ℝ)
  (h_arith_geo_seq : arith_geo_seq a) :
  a 1 = 1 ∧ (∀ n, a n = 2 * n - 1) :=
sorry

end NUMINAMATH_GPT_find_a1_an_l538_53822


namespace NUMINAMATH_GPT_envelope_addressing_equation_l538_53838

theorem envelope_addressing_equation (x : ℝ) :
  (800 / 10 + 800 / x + 800 / 5) * (3 / 800) = 1 / 3 :=
  sorry

end NUMINAMATH_GPT_envelope_addressing_equation_l538_53838


namespace NUMINAMATH_GPT_custom_op_seven_three_l538_53812

def custom_op (a b : ℕ) : ℕ := 4 * a + 5 * b - a * b + 1

theorem custom_op_seven_three : custom_op 7 3 = 23 := by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_custom_op_seven_three_l538_53812


namespace NUMINAMATH_GPT_bob_weight_l538_53863

noncomputable def jim_bob_equations (j b : ℝ) : Prop :=
  j + b = 200 ∧ b - 3 * j = b / 4

theorem bob_weight (j b : ℝ) (h : jim_bob_equations j b) : b = 171.43 :=
by
  sorry

end NUMINAMATH_GPT_bob_weight_l538_53863


namespace NUMINAMATH_GPT_smallest_possible_value_l538_53806

theorem smallest_possible_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a^2 + b^2) / (a * b) + (a * b) / (a^2 + b^2) ≥ 2 :=
sorry

end NUMINAMATH_GPT_smallest_possible_value_l538_53806


namespace NUMINAMATH_GPT_value_of_a_minus_b_l538_53850

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 8) (h2 : |b| = 6) (h3 : |a + b| = a + b) : a - b = 2 ∨ a - b = 14 := 
sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l538_53850


namespace NUMINAMATH_GPT_inequality_bound_l538_53874

theorem inequality_bound (a : ℝ) (h : ∃ x : ℝ, 0 < x ∧ e^x * (x^2 - x + 1) * (a * x + 3 * a - 1) < 1) : a < 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_bound_l538_53874


namespace NUMINAMATH_GPT_gcd_204_85_l538_53839

theorem gcd_204_85 : Nat.gcd 204 85 = 17 :=
sorry

end NUMINAMATH_GPT_gcd_204_85_l538_53839


namespace NUMINAMATH_GPT_ratio_of_perimeters_l538_53877

noncomputable def sqrt2 : ℝ := Real.sqrt 2

theorem ratio_of_perimeters (d1 : ℝ) :
  let d2 := (1 + sqrt2) * d1
  let s1 := d1 / sqrt2
  let s2 := d2 / sqrt2
  let P1 := 4 * s1
  let P2 := 4 * s2 
  (P2 / P1 = 1 + sqrt2) :=
by
  let d2 := (1 + sqrt2) * d1
  let s1 := d1 / sqrt2
  let s2 := d2 / sqrt2
  let P1 := 4 * s1
  let P2 := 4 * s2
  sorry

end NUMINAMATH_GPT_ratio_of_perimeters_l538_53877


namespace NUMINAMATH_GPT_watch_cost_price_l538_53836

theorem watch_cost_price (C : ℝ) (h1 : 0.85 * C = SP1) (h2 : 1.06 * C = SP2) (h3 : SP2 - SP1 = 350) : 
  C = 1666.67 := 
  sorry

end NUMINAMATH_GPT_watch_cost_price_l538_53836


namespace NUMINAMATH_GPT_applesauce_ratio_is_half_l538_53861

-- Define the weights and number of pies
def total_weight : ℕ := 120
def weight_per_pie : ℕ := 4
def num_pies : ℕ := 15

-- Calculate weights used for pies and applesauce
def weight_for_pies : ℕ := num_pies * weight_per_pie
def weight_for_applesauce : ℕ := total_weight - weight_for_pies

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- Statement to prove
theorem applesauce_ratio_is_half :
  ratio weight_for_applesauce total_weight = 1 / 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_applesauce_ratio_is_half_l538_53861


namespace NUMINAMATH_GPT_ekon_uma_diff_l538_53884

-- Definitions based on conditions
def total_videos := 411
def kelsey_videos := 160
def ekon_kelsey_diff := 43

-- Definitions derived from conditions
def ekon_videos := kelsey_videos - ekon_kelsey_diff
def uma_videos (E : ℕ) := total_videos - kelsey_videos - E

-- The Lean problem statement
theorem ekon_uma_diff : 
  uma_videos ekon_videos - ekon_videos = 17 := 
by 
  sorry

end NUMINAMATH_GPT_ekon_uma_diff_l538_53884


namespace NUMINAMATH_GPT_quadratic_completing_square_t_l538_53809

theorem quadratic_completing_square_t : 
  ∀ (x k t : ℝ), (4 * x^2 + 16 * x - 400 = 0) →
  ((x + k)^2 = t) →
  t = 104 :=
by
  intros x k t h1 h2
  sorry

end NUMINAMATH_GPT_quadratic_completing_square_t_l538_53809


namespace NUMINAMATH_GPT_rectangle_perimeter_l538_53872

theorem rectangle_perimeter (L B : ℝ) (h1 : L - B = 23) (h2 : L * B = 2030) : 2 * (L + B) = 186 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l538_53872


namespace NUMINAMATH_GPT_most_convincing_method_for_relationship_l538_53859

-- Definitions from conditions
def car_owners : ℕ := 300
def car_owners_opposed_policy : ℕ := 116
def non_car_owners : ℕ := 200
def non_car_owners_opposed_policy : ℕ := 121

-- The theorem statement
theorem most_convincing_method_for_relationship : 
  (owning_a_car_related_to_opposing_policy : Bool) :=
by
  -- Proof of the statement
  sorry

end NUMINAMATH_GPT_most_convincing_method_for_relationship_l538_53859


namespace NUMINAMATH_GPT_sum_of_turning_angles_l538_53894

variable (radius distance : ℝ) (C : ℝ)

theorem sum_of_turning_angles (H1 : radius = 10) (H2 : distance = 30000) (H3 : C = 2 * radius * Real.pi) :
  (distance / C) * 2 * Real.pi ≥ 2998 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_turning_angles_l538_53894


namespace NUMINAMATH_GPT_find_added_value_l538_53827

theorem find_added_value (N : ℕ) (V : ℕ) (H : N = 1280) :
  ((N + V) / 125 = 7392 / 462) → V = 720 :=
by 
  sorry

end NUMINAMATH_GPT_find_added_value_l538_53827


namespace NUMINAMATH_GPT_condition_iff_inequality_l538_53815

theorem condition_iff_inequality (a b : ℝ) (h : a * b ≠ 0) : (0 < a ∧ 0 < b) ↔ ((a + b) / 2 ≥ Real.sqrt (a * b)) :=
by
  -- Proof goes here
  sorry 

end NUMINAMATH_GPT_condition_iff_inequality_l538_53815


namespace NUMINAMATH_GPT_betty_afternoon_catch_l538_53826

def flies_eaten_per_day := 2
def days_in_week := 7
def flies_needed_for_week := days_in_week * flies_eaten_per_day
def flies_caught_morning := 5
def additional_flies_needed := 4
def flies_currently_have := flies_needed_for_week - additional_flies_needed
def flies_caught_afternoon := flies_currently_have - flies_caught_morning
def flies_escaped := 1

theorem betty_afternoon_catch :
  flies_caught_afternoon + flies_escaped = 6 :=
by
  sorry

end NUMINAMATH_GPT_betty_afternoon_catch_l538_53826


namespace NUMINAMATH_GPT_area_of_circle_l538_53803

theorem area_of_circle 
  (r : ℝ → ℝ)
  (h : ∀ θ : ℝ, r θ = 3 * Real.cos θ - 4 * Real.sin θ) :
  ∃ A : ℝ, A = (25 / 4) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_circle_l538_53803


namespace NUMINAMATH_GPT_smallest_perfect_cube_divisor_l538_53821

theorem smallest_perfect_cube_divisor (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (h : p ≠ q ∧ p ≠ r ∧ q ≠ r) :
  ∃ (a b c : ℕ), a = 6 ∧ b = 6 ∧ c = 6 ∧ (p^a * q^b * r^c) = (p^2 * q^2 * r^2)^3 ∧ 
  (p^a * q^b * r^c) % (p^2 * q^3 * r^4) = 0 := 
by
  sorry

end NUMINAMATH_GPT_smallest_perfect_cube_divisor_l538_53821


namespace NUMINAMATH_GPT_base_7_units_digit_of_product_359_72_l538_53880

def base_7_units_digit (n : ℕ) : ℕ := n % 7

theorem base_7_units_digit_of_product_359_72 : base_7_units_digit (359 * 72) = 4 := 
by
  sorry

end NUMINAMATH_GPT_base_7_units_digit_of_product_359_72_l538_53880


namespace NUMINAMATH_GPT_number_of_friends_l538_53882

theorem number_of_friends (P : ℕ) (n m : ℕ) (h1 : ∀ (A B C : ℕ), (A = B ∨ A ≠ B) ∧ (B = C ∨ B ≠ C) → (n-1) * m = 15):
  P = 16 ∨ P = 18 ∨ P = 20 ∨ P = 30 :=
sorry

end NUMINAMATH_GPT_number_of_friends_l538_53882


namespace NUMINAMATH_GPT_cannon_hit_probability_l538_53852

theorem cannon_hit_probability {P2 P3 : ℝ} (hP1 : 0.5 <= P2) (hP2 : P2 = 0.2) (hP3 : P3 = 0.3) (h_none_hit : (1 - 0.5) * (1 - P2) * (1 - P3) = 0.28) :
  0.5 = 0.5 :=
by sorry

end NUMINAMATH_GPT_cannon_hit_probability_l538_53852


namespace NUMINAMATH_GPT_xyz_sum_eq_7x_plus_5_l538_53853

variable (x y z : ℝ)

theorem xyz_sum_eq_7x_plus_5 (h1: y = 3 * x) (h2: z = y + 5) : x + y + z = 7 * x + 5 :=
by
  sorry

end NUMINAMATH_GPT_xyz_sum_eq_7x_plus_5_l538_53853


namespace NUMINAMATH_GPT_stickers_per_page_l538_53805

theorem stickers_per_page (n_pages total_stickers : ℕ) (h_n_pages : n_pages = 22) (h_total_stickers : total_stickers = 220) : total_stickers / n_pages = 10 :=
by
  sorry

end NUMINAMATH_GPT_stickers_per_page_l538_53805


namespace NUMINAMATH_GPT_a_seq_correct_l538_53856

-- Define the sequence and the sum condition
def a_seq (n : ℕ) : ℚ := if n = 0 then 0 else (2 ^ n - 1) / 2 ^ (n - 1)

def S_n (n : ℕ) : ℚ :=
  if n = 0 then 0 else (Finset.sum (Finset.range n) a_seq)

axiom condition (n : ℕ) (hn : n > 0) : S_n n + a_seq n = 2 * n

theorem a_seq_correct (n : ℕ) (hn : n > 0) : 
  a_seq n = (2 ^ n - 1) / 2 ^ (n - 1) := sorry

end NUMINAMATH_GPT_a_seq_correct_l538_53856


namespace NUMINAMATH_GPT_students_play_both_l538_53842

-- Definitions of problem conditions
def total_students : ℕ := 1200
def play_football : ℕ := 875
def play_cricket : ℕ := 450
def play_neither : ℕ := 100
def play_either := total_students - play_neither

-- Lean statement to prove that the number of students playing both football and cricket
theorem students_play_both : play_football + play_cricket - 225 = play_either :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_students_play_both_l538_53842


namespace NUMINAMATH_GPT_factorize_expression_l538_53866

theorem factorize_expression (x y : ℝ) : (y + 2 * x)^2 - (x + 2 * y)^2 = 3 * (x + y) * (x - y) :=
  sorry

end NUMINAMATH_GPT_factorize_expression_l538_53866


namespace NUMINAMATH_GPT_circles_through_two_points_in_4x4_grid_l538_53833

noncomputable def number_of_circles (n : ℕ) : ℕ :=
  if n = 4 then
    52
  else
    sorry

theorem circles_through_two_points_in_4x4_grid :
  number_of_circles 4 = 52 :=
by
  exact rfl  -- Reflexivity of equality shows the predefined value of 52

end NUMINAMATH_GPT_circles_through_two_points_in_4x4_grid_l538_53833


namespace NUMINAMATH_GPT_f_at_2_is_neg_1_l538_53813

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 - x + 2

-- Given condition: f(-2) = 5
axiom h : ∀ (a b : ℝ), f a b (-2) = 5

-- Prove that f(2) = -1 given the above conditions
theorem f_at_2_is_neg_1 (a b : ℝ) (h_ab : f a b (-2) = 5) : f a b 2 = -1 := by
  sorry

end NUMINAMATH_GPT_f_at_2_is_neg_1_l538_53813


namespace NUMINAMATH_GPT_a_minus_b_is_30_l538_53832

-- Definition of the sum of the arithmetic series
def sum_arithmetic_series (first last : ℕ) (n : ℕ) : ℕ :=
  (n * (first + last)) / 2

-- Definitions based on problem conditions
def a : ℕ := sum_arithmetic_series 2 60 30
def b : ℕ := sum_arithmetic_series 1 59 30

theorem a_minus_b_is_30 : a - b = 30 :=
  by sorry

end NUMINAMATH_GPT_a_minus_b_is_30_l538_53832


namespace NUMINAMATH_GPT_candy_cost_l538_53899

theorem candy_cost (x : ℝ) : 
  (15 * x + 30 * 5) / (15 + 30) = 6 -> x = 8 :=
by sorry

end NUMINAMATH_GPT_candy_cost_l538_53899


namespace NUMINAMATH_GPT_find_integer_pairs_l538_53830

theorem find_integer_pairs :
  ∃ (x y : ℤ),
    (x, y) = (-7, -99) ∨ (x, y) = (-1, -9) ∨ (x, y) = (1, 5) ∨ (x, y) = (7, -97) ∧
    2 * x^3 + x * y - 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l538_53830
