import Mathlib

namespace NUMINAMATH_CALUDE_complex_magnitude_l3468_346864

theorem complex_magnitude (z : ℂ) (h : (2 - I) * z = 6 + 2 * I) : Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3468_346864


namespace NUMINAMATH_CALUDE_john_annual_maintenance_expenses_l3468_346895

/-- Represents John's annual car maintenance expenses --/
def annual_maintenance_expenses (
  annual_mileage : ℕ)
  (oil_change_interval : ℕ)
  (free_oil_changes : ℕ)
  (oil_change_cost : ℕ)
  (tire_rotation_interval : ℕ)
  (tire_rotation_cost : ℕ)
  (brake_pad_interval : ℕ)
  (brake_pad_cost : ℕ) : ℕ :=
  let paid_oil_changes := annual_mileage / oil_change_interval - free_oil_changes
  let annual_oil_change_cost := paid_oil_changes * oil_change_cost
  let annual_tire_rotation_cost := (annual_mileage / tire_rotation_interval) * tire_rotation_cost
  let annual_brake_pad_cost := (annual_mileage * brake_pad_cost) / brake_pad_interval
  annual_oil_change_cost + annual_tire_rotation_cost + annual_brake_pad_cost

/-- Theorem stating John's annual maintenance expenses --/
theorem john_annual_maintenance_expenses :
  annual_maintenance_expenses 12000 3000 1 50 6000 40 24000 200 = 330 := by
  sorry

end NUMINAMATH_CALUDE_john_annual_maintenance_expenses_l3468_346895


namespace NUMINAMATH_CALUDE_circle_intersection_ratio_l3468_346847

theorem circle_intersection_ratio (m : ℝ) (h : 0 < m ∧ m < 1) :
  let R : ℝ := 1  -- We can set R = 1 without loss of generality
  let common_area := 2 * R^2 * (Real.arccos m - m * Real.sqrt (1 - m^2))
  let third_circle_area := π * (m * R)^2
  common_area / third_circle_area = 2 * (Real.arccos m - m * Real.sqrt (1 - m^2)) / (π * m^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_ratio_l3468_346847


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l3468_346813

theorem green_shirt_pairs (blue_students : ℕ) (green_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_students = 65 →
  green_students = 95 →
  total_students = 160 →
  total_pairs = 80 →
  blue_blue_pairs = 25 →
  blue_students + green_students = total_students →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l3468_346813


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3468_346814

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 5 * n % 26 = 2024 % 26 ∧ ∀ (m : ℕ), m > 0 ∧ m < n → 5 * m % 26 ≠ 2024 % 26 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3468_346814


namespace NUMINAMATH_CALUDE_coffee_beans_per_cup_l3468_346860

/-- Represents the coffee consumption and cost scenario for Maddie's mom --/
structure CoffeeScenario where
  cups_per_day : ℕ
  coffee_bag_cost : ℚ
  coffee_bag_ounces : ℚ
  milk_gallons_per_week : ℚ
  milk_cost_per_gallon : ℚ
  weekly_coffee_expense : ℚ

/-- Calculates the ounces of coffee beans per cup --/
def ounces_per_cup (scenario : CoffeeScenario) : ℚ :=
  sorry

/-- Theorem stating that the ounces of coffee beans per cup is 1.5 --/
theorem coffee_beans_per_cup (scenario : CoffeeScenario) 
  (h1 : scenario.cups_per_day = 2)
  (h2 : scenario.coffee_bag_cost = 8)
  (h3 : scenario.coffee_bag_ounces = 21/2)
  (h4 : scenario.milk_gallons_per_week = 1/2)
  (h5 : scenario.milk_cost_per_gallon = 4)
  (h6 : scenario.weekly_coffee_expense = 18) :
  ounces_per_cup scenario = 3/2 :=
sorry

end NUMINAMATH_CALUDE_coffee_beans_per_cup_l3468_346860


namespace NUMINAMATH_CALUDE_ball_max_height_l3468_346881

/-- The height function of the ball's trajectory -/
def h (t : ℝ) : ℝ := -16 * t^2 + 32 * t + 15

/-- Theorem stating that the maximum height of the ball is 31 feet -/
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 31 :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l3468_346881


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3468_346858

/-- The general term of the sequence -/
def sequenceTerm (n : ℕ) : ℚ := (2 * n) / (2 * n + 1)

/-- The 10th term of the sequence is 20/21 -/
theorem tenth_term_of_sequence : sequenceTerm 10 = 20 / 21 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3468_346858


namespace NUMINAMATH_CALUDE_digit_120th_of_7_26th_l3468_346854

theorem digit_120th_of_7_26th : ∃ (seq : ℕ → ℕ), 
  (∀ n, seq n < 10) ∧ 
  (∀ n, seq (n + 9) = seq n) ∧
  (∀ n, (7 * 10^n) % 26 = (seq n * 10^8 + seq (n+1) * 10^7 + seq (n+2) * 10^6 + 
                           seq (n+3) * 10^5 + seq (n+4) * 10^4 + seq (n+5) * 10^3 + 
                           seq (n+6) * 10^2 + seq (n+7) * 10 + seq (n+8)) % 26) ∧
  seq 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_120th_of_7_26th_l3468_346854


namespace NUMINAMATH_CALUDE_class_ratio_theorem_l3468_346824

-- Define the class structure
structure ClassComposition where
  total_students : ℕ
  girls : ℕ
  boys : ℕ

-- Define the condition given in the problem
def satisfies_condition (c : ClassComposition) : Prop :=
  2 * c.girls * 5 = 3 * c.total_students

-- Define the property we want to prove
def has_correct_ratio (c : ClassComposition) : Prop :=
  7 * c.girls = 3 * c.boys

-- The theorem to prove
theorem class_ratio_theorem (c : ClassComposition) 
  (h1 : c.total_students = c.girls + c.boys)
  (h2 : satisfies_condition c) : 
  has_correct_ratio c := by
  sorry

#check class_ratio_theorem

end NUMINAMATH_CALUDE_class_ratio_theorem_l3468_346824


namespace NUMINAMATH_CALUDE_victors_journey_l3468_346832

/-- The distance from Victor's home to the airport --/
def s : ℝ := 240

/-- Victor's initial speed --/
def initial_speed : ℝ := 60

/-- Victor's increased speed --/
def increased_speed : ℝ := 80

/-- Time spent at initial speed --/
def initial_time : ℝ := 0.5

/-- Time difference if Victor continued at initial speed --/
def late_time : ℝ := 0.25

/-- Time difference after increasing speed --/
def early_time : ℝ := 0.25

theorem victors_journey :
  ∃ (t : ℝ),
    s = initial_speed * initial_time + initial_speed * (t + late_time) ∧
    s = initial_speed * initial_time + increased_speed * (t - early_time) :=
by sorry

end NUMINAMATH_CALUDE_victors_journey_l3468_346832


namespace NUMINAMATH_CALUDE_min_value_of_difference_l3468_346878

theorem min_value_of_difference (x y z : ℝ) : 
  0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ 4 →
  y^2 - x^2 = 2 →
  z^2 - y^2 = 2 →
  (2 * (2 - Real.sqrt 3) : ℝ) ≤ |x - y| + |y - z| ∧ |x - y| + |y - z| ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_difference_l3468_346878


namespace NUMINAMATH_CALUDE_toaster_sales_at_promo_price_l3468_346806

-- Define the inverse proportionality constant
def k : ℝ := 15 * 600

-- Define the original price and number of customers
def original_price : ℝ := 600
def original_customers : ℝ := 15

-- Define the promotional price
def promo_price : ℝ := 450

-- Define the additional sales increase factor
def promo_factor : ℝ := 1.1

-- Theorem statement
theorem toaster_sales_at_promo_price :
  let normal_sales := k / promo_price
  let promo_sales := normal_sales * promo_factor
  promo_sales = 22 := by sorry

end NUMINAMATH_CALUDE_toaster_sales_at_promo_price_l3468_346806


namespace NUMINAMATH_CALUDE_angle_with_special_supplement_complement_l3468_346894

theorem angle_with_special_supplement_complement : ∃ (x : ℝ), 
  (0 < x) ∧ (x < 180) ∧ (180 - x = 4 * (90 - x)) ∧ (x = 60) := by
  sorry

end NUMINAMATH_CALUDE_angle_with_special_supplement_complement_l3468_346894


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l3468_346835

/-- The trajectory of point G -/
def trajectory (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- The condition that the product of slopes of GE and FG is -4 -/
def slope_condition (x y : ℝ) : Prop := 
  y ≠ 0 → (y / (x - 1)) * (y / (x + 1)) = -4

/-- The line passing through (0, -1) with slope k -/
def line (k x : ℝ) : ℝ := k * x - 1

/-- The x-coordinates of the intersection points sum to 8 -/
def intersection_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    trajectory x₁ (line k x₁) ∧ 
    trajectory x₂ (line k x₂) ∧ 
    x₁ + x₂ = 8

theorem trajectory_and_intersection :
  (∀ x y : ℝ, slope_condition x y → trajectory x y) ∧
  (∀ k : ℝ, intersection_condition k → k = 2) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l3468_346835


namespace NUMINAMATH_CALUDE_complex_on_line_l3468_346885

theorem complex_on_line (a : ℝ) : 
  (∃ z : ℂ, z = (a - Complex.I) / (1 + Complex.I) ∧ 
   z.re - z.im + 1 = 0) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_line_l3468_346885


namespace NUMINAMATH_CALUDE_loose_coins_amount_l3468_346884

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def bills_given : ℕ := 40
def change_received : ℕ := 10

theorem loose_coins_amount : 
  (flour_cost + cake_stand_cost + change_received) - bills_given = 3 := by
  sorry

end NUMINAMATH_CALUDE_loose_coins_amount_l3468_346884


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3468_346805

def f (a x : ℝ) := -x^2 + 2*a*x - 3

theorem quadratic_function_properties :
  (∃ x : ℝ, -3 < x ∧ x < -1 ∧ f (-2) x < 0) ∧
  (∀ x : ℝ, x ∈ Set.Icc 1 5 → ∀ a : ℝ, a > -2 * Real.sqrt 3 → f a x < 3 * a * x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3468_346805


namespace NUMINAMATH_CALUDE_rented_movie_cost_l3468_346853

theorem rented_movie_cost (ticket_price : ℚ) (num_tickets : ℕ) (bought_movie_price : ℚ) (total_spent : ℚ) :
  ticket_price = 10.62 →
  num_tickets = 2 →
  bought_movie_price = 13.95 →
  total_spent = 36.78 →
  total_spent - (ticket_price * num_tickets + bought_movie_price) = 1.59 :=
by sorry

end NUMINAMATH_CALUDE_rented_movie_cost_l3468_346853


namespace NUMINAMATH_CALUDE_max_pieces_on_chessboard_l3468_346850

/-- Represents a chessboard with red and blue pieces -/
structure Chessboard :=
  (size : Nat)
  (red_pieces : Finset (Nat × Nat))
  (blue_pieces : Finset (Nat × Nat))

/-- Counts the number of pieces of the opposite color that a piece can see -/
def count_opposite_color (board : Chessboard) (pos : Nat × Nat) (is_red : Bool) : Nat :=
  sorry

/-- Checks if the chessboard configuration is valid -/
def is_valid_configuration (board : Chessboard) : Prop :=
  board.size = 200 ∧
  (∀ pos ∈ board.red_pieces, count_opposite_color board pos true = 5) ∧
  (∀ pos ∈ board.blue_pieces, count_opposite_color board pos false = 5)

/-- The main theorem stating the maximum number of pieces on the chessboard -/
theorem max_pieces_on_chessboard (board : Chessboard) :
  is_valid_configuration board →
  Finset.card board.red_pieces + Finset.card board.blue_pieces ≤ 3800 :=
sorry

end NUMINAMATH_CALUDE_max_pieces_on_chessboard_l3468_346850


namespace NUMINAMATH_CALUDE_grade_calculation_l3468_346848

/-- Represents the weighted average calculation for a student's grades --/
def weighted_average (math history science geography : ℝ) : ℝ :=
  0.3 * math + 0.3 * history + 0.2 * science + 0.2 * geography

/-- Theorem stating the conditions and the result to be proven --/
theorem grade_calculation (math history science geography : ℝ) :
  math = 74 →
  history = 81 →
  science = geography + 5 →
  science ≥ 75 →
  science = 86.25 →
  geography = 81.25 →
  weighted_average math history science geography = 80 :=
by
  sorry

#eval weighted_average 74 81 86.25 81.25

end NUMINAMATH_CALUDE_grade_calculation_l3468_346848


namespace NUMINAMATH_CALUDE_initial_number_proof_l3468_346803

theorem initial_number_proof (N : ℕ) : 
  (∀ k < 5, ¬ (23 ∣ (N + k))) → 
  (23 ∣ (N + 5)) → 
  N = 18 :=
by sorry

end NUMINAMATH_CALUDE_initial_number_proof_l3468_346803


namespace NUMINAMATH_CALUDE_non_foreign_male_students_l3468_346892

theorem non_foreign_male_students 
  (total_students : ℕ) 
  (female_ratio : ℚ) 
  (foreign_male_ratio : ℚ) :
  total_students = 300 →
  female_ratio = 2/3 →
  foreign_male_ratio = 1/10 →
  (total_students : ℚ) * (1 - female_ratio) * (1 - foreign_male_ratio) = 90 := by
  sorry

end NUMINAMATH_CALUDE_non_foreign_male_students_l3468_346892


namespace NUMINAMATH_CALUDE_scenario_proof_l3468_346879

theorem scenario_proof (a b c d : ℝ) 
  (h1 : a * b * c * d > 0) 
  (h2 : a < c) 
  (h3 : b * c * d < 0) : 
  a < 0 ∧ b > 0 ∧ c < 0 ∧ d > 0 :=
by sorry

end NUMINAMATH_CALUDE_scenario_proof_l3468_346879


namespace NUMINAMATH_CALUDE_rowing_time_ratio_l3468_346833

theorem rowing_time_ratio (man_speed : ℝ) (current_speed : ℝ) 
  (h1 : man_speed = 3.3) (h2 : current_speed = 1.1) :
  (man_speed - current_speed) / (man_speed + current_speed) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rowing_time_ratio_l3468_346833


namespace NUMINAMATH_CALUDE_largest_number_l3468_346808

theorem largest_number (a b c d : ℝ) (h1 : a = 1/2) (h2 : b = -1) (h3 : c = |(-2)|) (h4 : d = -3) :
  c ≥ a ∧ c ≥ b ∧ c ≥ d :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l3468_346808


namespace NUMINAMATH_CALUDE_smallest_d_for_perfect_square_l3468_346851

theorem smallest_d_for_perfect_square : ∃ (n : ℕ), 
  14 * 3150 = n^2 ∧ 
  ∀ (d : ℕ), d > 0 ∧ d < 14 → ¬∃ (m : ℕ), d * 3150 = m^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_for_perfect_square_l3468_346851


namespace NUMINAMATH_CALUDE_min_pumps_for_given_reservoir_l3468_346831

/-- Represents the characteristics of a reservoir with a leakage problem -/
structure Reservoir where
  single_pump_time : ℝ
  double_pump_time : ℝ
  target_time : ℝ

/-- Calculates the minimum number of pumps needed to fill the reservoir within the target time -/
def min_pumps_needed (r : Reservoir) : ℕ :=
  sorry

/-- Theorem stating that for the given reservoir conditions, at least 3 pumps are needed -/
theorem min_pumps_for_given_reservoir :
  let r : Reservoir := {
    single_pump_time := 8,
    double_pump_time := 3.2,
    target_time := 2
  }
  min_pumps_needed r = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_pumps_for_given_reservoir_l3468_346831


namespace NUMINAMATH_CALUDE_unique_triples_l3468_346883

theorem unique_triples : 
  ∀ (a b c : ℕ+), 
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
    (∃ (k₁ : ℕ+), (2 * a - 1 : ℤ) = k₁ * b) →
    (∃ (k₂ : ℕ+), (2 * b - 1 : ℤ) = k₂ * c) →
    (∃ (k₃ : ℕ+), (2 * c - 1 : ℤ) = k₃ * a) →
    ((a = 7 ∧ b = 13 ∧ c = 25) ∨
     (a = 13 ∧ b = 25 ∧ c = 7) ∨
     (a = 25 ∧ b = 7 ∧ c = 13)) :=
by sorry

end NUMINAMATH_CALUDE_unique_triples_l3468_346883


namespace NUMINAMATH_CALUDE_square_characterization_l3468_346875

theorem square_characterization (A : ℕ) :
  (∃ k : ℕ, A = k^2) ↔
  (∀ n : ℕ, ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ n ∣ ((A + i)^2 - A)) :=
by sorry

end NUMINAMATH_CALUDE_square_characterization_l3468_346875


namespace NUMINAMATH_CALUDE_triangle_properties_l3468_346838

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 - t.c^2 = t.b^2 - (8 * t.b * t.c) / 5)
  (h2 : t.a = 6)
  (h3 : Real.sin t.B = 4/5) :
  (Real.sin t.A = 3/5) ∧ 
  ((1/2 * t.b * t.c * Real.sin t.A = 24) ∨ 
   (1/2 * t.b * t.c * Real.sin t.A = 168/25)) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3468_346838


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l3468_346897

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 4) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 61 / 64 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l3468_346897


namespace NUMINAMATH_CALUDE_f_positive_iff_x_range_l3468_346866

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

-- State the theorem
theorem f_positive_iff_x_range (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  (∀ x : ℝ, f a x > 0) ↔ (∀ x : ℝ, x < 1 ∨ x > 3) :=
sorry

end NUMINAMATH_CALUDE_f_positive_iff_x_range_l3468_346866


namespace NUMINAMATH_CALUDE_stephanie_oranges_l3468_346886

theorem stephanie_oranges (store_visits : ℕ) (oranges_per_visit : ℕ) : 
  store_visits = 8 → oranges_per_visit = 2 → store_visits * oranges_per_visit = 16 := by
sorry

end NUMINAMATH_CALUDE_stephanie_oranges_l3468_346886


namespace NUMINAMATH_CALUDE_no_solution_range_l3468_346836

theorem no_solution_range (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 5| > m) ↔ m < 6 := by sorry

end NUMINAMATH_CALUDE_no_solution_range_l3468_346836


namespace NUMINAMATH_CALUDE_runner_distance_at_click_l3468_346802

/-- The time in seconds for which the camera timer is set -/
def timer_setting : ℝ := 45

/-- The runner's speed in yards per second -/
def runner_speed : ℝ := 10

/-- The speed of sound in feet per second without headwind -/
def sound_speed : ℝ := 1100

/-- The reduction factor of sound speed due to headwind -/
def sound_speed_reduction : ℝ := 0.1

/-- The effective speed of sound in feet per second with headwind -/
def effective_sound_speed : ℝ := sound_speed * (1 - sound_speed_reduction)

/-- The distance the runner travels in feet at time t -/
def runner_distance (t : ℝ) : ℝ := runner_speed * 3 * t

/-- The distance sound travels in feet at time t after the camera click -/
def sound_distance (t : ℝ) : ℝ := effective_sound_speed * (t - timer_setting)

/-- The time when the runner hears the camera click -/
noncomputable def hearing_time : ℝ := 
  (effective_sound_speed * timer_setting) / (effective_sound_speed - runner_speed * 3)

theorem runner_distance_at_click : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |runner_distance hearing_time / 3 - 464| < ε :=
sorry

end NUMINAMATH_CALUDE_runner_distance_at_click_l3468_346802


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l3468_346837

theorem min_product_of_three_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y → 
  x * y * z ≥ 1/18 := by
  sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l3468_346837


namespace NUMINAMATH_CALUDE_grill_coal_consumption_l3468_346842

theorem grill_coal_consumption (total_time : ℕ) (bags : ℕ) (coals_per_bag : ℕ) 
  (h1 : total_time = 240)
  (h2 : bags = 3)
  (h3 : coals_per_bag = 60) :
  (bags * coals_per_bag) / (total_time / 20) = 15 := by
  sorry

end NUMINAMATH_CALUDE_grill_coal_consumption_l3468_346842


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3468_346841

/-- Given a parabola and a line intersecting it, prove the value of m -/
theorem parabola_line_intersection (x₁ x₂ y₁ y₂ m : ℝ) : 
  (x₁^2 = 4*y₁) →  -- Point A on parabola
  (x₂^2 = 4*y₂) →  -- Point B on parabola
  (∃ k, y₁ = k*x₁ + m ∧ y₂ = k*x₂ + m) →  -- Line equation
  (x₁ * x₂ = -4) →  -- Product of x-coordinates
  (m = 1) := by
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3468_346841


namespace NUMINAMATH_CALUDE_negation_of_forall_inequality_l3468_346862

theorem negation_of_forall_inequality (x : ℝ) :
  ¬(∀ x > 1, x - 1 > Real.log x) ↔ ∃ x > 1, x - 1 ≤ Real.log x :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_inequality_l3468_346862


namespace NUMINAMATH_CALUDE_climb_10_steps_in_8_moves_l3468_346809

/-- The number of ways to climb n steps in exactly k moves, where each move can be either 1 or 2 steps. -/
def climbWays (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The theorem states that there are 28 ways to climb 10 steps in exactly 8 moves. -/
theorem climb_10_steps_in_8_moves : climbWays 10 8 = 28 := by sorry

end NUMINAMATH_CALUDE_climb_10_steps_in_8_moves_l3468_346809


namespace NUMINAMATH_CALUDE_probability_letter_in_mathematical_l3468_346820

def alphabet : Finset Char := sorry

def mathematical : String := "MATHEMATICAL"

theorem probability_letter_in_mathematical :
  let unique_letters := mathematical.toList.toFinset
  (unique_letters.card : ℚ) / (alphabet.card : ℚ) = 4 / 13 := by sorry

end NUMINAMATH_CALUDE_probability_letter_in_mathematical_l3468_346820


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3468_346876

def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3468_346876


namespace NUMINAMATH_CALUDE_lisa_interest_earned_l3468_346844

/-- UltraSavingsAccount represents the parameters of the savings account -/
structure UltraSavingsAccount where
  principal : ℝ
  rate : ℝ
  years : ℕ

/-- calculate_interest computes the interest earned for a given UltraSavingsAccount -/
def calculate_interest (account : UltraSavingsAccount) : ℝ :=
  account.principal * ((1 + account.rate) ^ account.years - 1)

/-- Theorem stating that Lisa's interest earned is $821 -/
theorem lisa_interest_earned (account : UltraSavingsAccount) 
  (h1 : account.principal = 2000)
  (h2 : account.rate = 0.035)
  (h3 : account.years = 10) :
  ⌊calculate_interest account⌋ = 821 := by
  sorry

end NUMINAMATH_CALUDE_lisa_interest_earned_l3468_346844


namespace NUMINAMATH_CALUDE_difference_of_numbers_l3468_346863

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 45) (h2 : x * y = 504) : 
  |x - y| = 3 := by sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l3468_346863


namespace NUMINAMATH_CALUDE_exists_expression_equal_100_l3468_346887

/-- An arithmetic expression using numbers 1 to 9 --/
inductive Expr
  | num : Fin 9 → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression --/
def eval : Expr → ℚ
  | Expr.num n => n.val + 1
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Check if an expression uses each number from 1 to 9 exactly once --/
def usesAllNumbers : Expr → Bool := sorry

/-- Theorem: There exists an arithmetic expression using numbers 1 to 9 that evaluates to 100 --/
theorem exists_expression_equal_100 : 
  ∃ e : Expr, usesAllNumbers e ∧ eval e = 100 := by sorry

end NUMINAMATH_CALUDE_exists_expression_equal_100_l3468_346887


namespace NUMINAMATH_CALUDE_sum_a_d_l3468_346801

theorem sum_a_d (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 42)
  (h2 : b + c = 6) : 
  a + d = 7 := by sorry

end NUMINAMATH_CALUDE_sum_a_d_l3468_346801


namespace NUMINAMATH_CALUDE_range_of_a_l3468_346890

def A (a : ℝ) := {x : ℝ | 1 ≤ x ∧ x ≤ a}

def B (a : ℝ) := {y : ℝ | ∃ x ∈ A a, y = 5*x - 6}

def C (a : ℝ) := {m : ℝ | ∃ x ∈ A a, m = x^2}

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ A a → (1 ≤ x ∧ x ≤ a)) → 
  (∀ y : ℝ, y ∈ B a ↔ ∃ x ∈ A a, y = 5*x - 6) → 
  (∀ m : ℝ, m ∈ C a ↔ ∃ x ∈ A a, m = x^2) → 
  (B a ∩ C a = C a) → 
  (2 ≤ a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3468_346890


namespace NUMINAMATH_CALUDE_initial_population_theorem_l3468_346846

def village_population (P : ℕ) : Prop :=
  ⌊(P : ℝ) * 0.95 * 0.80⌋ = 3553

theorem initial_population_theorem :
  ∃ P : ℕ, village_population P ∧ P ≥ 4678 ∧ P < 4679 :=
sorry

end NUMINAMATH_CALUDE_initial_population_theorem_l3468_346846


namespace NUMINAMATH_CALUDE_special_ellipse_properties_l3468_346818

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  P : ℝ × ℝ
  h_P_on_ellipse : (P.1 / a) ^ 2 + (P.2 / b) ^ 2 = 1
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  h_PF₂_eq_F₁F₂ : dist P F₂ = dist F₁ F₂
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_AB_on_ellipse : (A.1 / a) ^ 2 + (A.2 / b) ^ 2 = 1 ∧ (B.1 / a) ^ 2 + (B.2 / b) ^ 2 = 1
  h_AB_on_PF₂ : ∃ (t : ℝ), A = (1 - t) • P + t • F₂ ∧ B = (1 - t) • P + t • F₂
  M : ℝ × ℝ
  N : ℝ × ℝ
  h_MN_on_circle : (M.1 + 1) ^ 2 + (M.2 - Real.sqrt 3) ^ 2 = 16 ∧
                   (N.1 + 1) ^ 2 + (N.2 - Real.sqrt 3) ^ 2 = 16
  h_MN_on_PF₂ : ∃ (s : ℝ), M = (1 - s) • P + s • F₂ ∧ N = (1 - s) • P + s • F₂
  h_MN_AB_ratio : dist M N = (5 / 8) * dist A B

/-- The eccentricity and equation of the special ellipse -/
theorem special_ellipse_properties (e : SpecialEllipse) :
  (∃ (c : ℝ), c > 0 ∧ c ^ 2 = a ^ 2 - b ^ 2 ∧ c / a = 1 / 2) ∧
  (∃ (k : ℝ), k > 0 ∧ e.a ^ 2 = 16 * k ∧ e.b ^ 2 = 12 * k) :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_properties_l3468_346818


namespace NUMINAMATH_CALUDE_expression_simplification_l3468_346898

theorem expression_simplification (a b : ℝ) (h1 : a > b) (h2 : b > 1) :
  (a^(b+1) * b^(a+1)) / (b^(b+1) * a^(a+1)) = (a/b)^(b-a) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3468_346898


namespace NUMINAMATH_CALUDE_four_digit_number_fraction_l3468_346815

theorem four_digit_number_fraction (a b c d : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) →  -- Ensuring each digit is less than 10
  (∃ k : ℚ, a = k * b) →  -- First digit is a fraction of the second
  (c = a + b) →  -- Third digit is the sum of first and second
  (d = 3 * b) →  -- Last digit is 3 times the second
  (1000 * a + 100 * b + 10 * c + d = 1349) →  -- The number is 1349
  (a : ℚ) / b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_four_digit_number_fraction_l3468_346815


namespace NUMINAMATH_CALUDE_inequality_theorem_l3468_346872

theorem inequality_theorem (m n : ℕ) (h : m > n) :
  (1 + 1 / m : ℝ) ^ m > (1 + 1 / n : ℝ) ^ n ∧
  (1 + 1 / m : ℝ) ^ (m + 1) < (1 + 1 / n : ℝ) ^ (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3468_346872


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3468_346849

theorem quadratic_equation_solutions (a b : ℝ) :
  (∀ x : ℝ, x = -1 ∨ x = 2 → -a * x^2 + b * x = -2) →
  (-a * (-1)^2 + b * (-1) + 2 = 0) ∧ (-a * 2^2 + b * 2 + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3468_346849


namespace NUMINAMATH_CALUDE_exists_y_d_satisfying_equation_l3468_346889

theorem exists_y_d_satisfying_equation : ∃ (y d : ℕ), 3^y + 6*d = 735 := by sorry


end NUMINAMATH_CALUDE_exists_y_d_satisfying_equation_l3468_346889


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3468_346891

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y : ℝ, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3468_346891


namespace NUMINAMATH_CALUDE_unique_k_for_equation_l3468_346843

theorem unique_k_for_equation : ∃! k : ℕ+, 
  (∃ a b : ℕ+, a^2 + b^2 = k * a * b) ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_for_equation_l3468_346843


namespace NUMINAMATH_CALUDE_toms_incorrect_calculation_correct_calculation_l3468_346868

/-- The original number Tom was working with -/
def y : ℤ := 114

/-- Tom's incorrect calculation -/
theorem toms_incorrect_calculation : (y - 14) / 2 = 50 := by sorry

/-- The correct calculation -/
theorem correct_calculation : ((y - 5) / 7 : ℚ).floor = 15 := by sorry

end NUMINAMATH_CALUDE_toms_incorrect_calculation_correct_calculation_l3468_346868


namespace NUMINAMATH_CALUDE_sine_inequality_l3468_346870

theorem sine_inequality (n : ℕ) (x : ℝ) : 
  Real.sin x * (n * Real.sin x - Real.sin (n * x)) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l3468_346870


namespace NUMINAMATH_CALUDE_initial_tickets_correct_l3468_346810

/-- The number of tickets Adam initially bought at the fair -/
def initial_tickets : ℕ := 13

/-- The number of tickets left after riding the ferris wheel -/
def tickets_left : ℕ := 4

/-- The cost of each ticket in dollars -/
def ticket_cost : ℕ := 9

/-- The total amount spent on the ferris wheel in dollars -/
def ferris_wheel_cost : ℕ := 81

/-- Theorem stating that the initial number of tickets is correct -/
theorem initial_tickets_correct : 
  initial_tickets = (ferris_wheel_cost / ticket_cost) + tickets_left := by
  sorry

end NUMINAMATH_CALUDE_initial_tickets_correct_l3468_346810


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3468_346817

/-- An arithmetic progression of positive integers -/
def arithmetic_progression (s : ℕ → ℕ) : Prop :=
  ∃ a d : ℕ, ∀ n : ℕ, s n = a + (n - 1) * d

theorem arithmetic_progression_x_value
  (s : ℕ → ℕ) (x : ℝ)
  (h_arithmetic : arithmetic_progression s)
  (h_s1 : s (s 1) = x + 2)
  (h_s2 : s (s 2) = x^2 + 18)
  (h_s3 : s (s 3) = 2*x^2 + 18) :
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3468_346817


namespace NUMINAMATH_CALUDE_lucas_units_digit_l3468_346871

-- Define Lucas numbers
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

-- Theorem statement
theorem lucas_units_digit :
  unitsDigit (lucas (lucas 9)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_lucas_units_digit_l3468_346871


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_ratio_l3468_346855

theorem isosceles_right_triangle_ratio (a c : ℝ) : 
  a > 0 → -- Ensure a is positive
  c^2 = 2 * a^2 → -- Pythagorean theorem for isosceles right triangle
  (2 * a) / c = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_ratio_l3468_346855


namespace NUMINAMATH_CALUDE_money_left_calculation_l3468_346823

/-- The amount of money John has left after buying pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let medium_pizza_cost := 3 * q
  let large_pizza_cost := 4 * q
  let total_spent := 4 * drink_cost + medium_pizza_cost + 2 * large_pizza_cost
  50 - total_spent

/-- Theorem stating that the money left is 50 - 15q -/
theorem money_left_calculation (q : ℝ) : money_left q = 50 - 15 * q := by
  sorry

end NUMINAMATH_CALUDE_money_left_calculation_l3468_346823


namespace NUMINAMATH_CALUDE_cube_vertex_shapes_l3468_346867

-- Define a cube
structure Cube where
  vertices : Fin 8 → Point3D

-- Define a selection of 4 vertices from a cube
def VertexSelection (c : Cube) := Fin 4 → Fin 8

-- Define geometric shapes that can be formed by 4 vertices
inductive Shape
  | Rectangle
  | TetrahedronIsoscelesRight
  | TetrahedronEquilateral
  | TetrahedronRight

-- Function to check if a selection of vertices forms a specific shape
def formsShape (c : Cube) (s : VertexSelection c) (shape : Shape) : Prop :=
  match shape with
  | Shape.Rectangle => sorry
  | Shape.TetrahedronIsoscelesRight => sorry
  | Shape.TetrahedronEquilateral => sorry
  | Shape.TetrahedronRight => sorry

-- Theorem stating that all these shapes can be formed by selecting 4 vertices from a cube
theorem cube_vertex_shapes (c : Cube) :
  ∃ (s₁ s₂ s₃ s₄ : VertexSelection c),
    formsShape c s₁ Shape.Rectangle ∧
    formsShape c s₂ Shape.TetrahedronIsoscelesRight ∧
    formsShape c s₃ Shape.TetrahedronEquilateral ∧
    formsShape c s₄ Shape.TetrahedronRight :=
  sorry

end NUMINAMATH_CALUDE_cube_vertex_shapes_l3468_346867


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_sufficient_condition_range_l3468_346807

-- Part I
theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 3*a*x + 9 > 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

-- Part II
theorem sufficient_condition_range (m : ℝ) :
  ((∀ x : ℝ, x^2 + 2*x - 8 < 0 → x - m > 0) ∧
   (∃ x : ℝ, x - m > 0 ∧ x^2 + 2*x - 8 ≥ 0)) →
  m ≤ -4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_sufficient_condition_range_l3468_346807


namespace NUMINAMATH_CALUDE_greatest_gcd_square_successor_l3468_346800

theorem greatest_gcd_square_successor (n : ℕ+) : 
  ∃ (k : ℕ+), Nat.gcd (6 * n^2) (n + 1) ≤ 6 ∧ 
  Nat.gcd (6 * k^2) (k + 1) = 6 :=
sorry

end NUMINAMATH_CALUDE_greatest_gcd_square_successor_l3468_346800


namespace NUMINAMATH_CALUDE_village_population_l3468_346821

theorem village_population (population : ℕ) : 
  (90 : ℚ) / 100 * population = 8100 → population = 9000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3468_346821


namespace NUMINAMATH_CALUDE_number_of_boys_l3468_346804

theorem number_of_boys (total_pupils : ℕ) (girls : ℕ) (teachers : ℕ) 
  (h1 : total_pupils = 626) 
  (h2 : girls = 308) 
  (h3 : teachers = 36) : 
  total_pupils - girls - teachers = 282 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l3468_346804


namespace NUMINAMATH_CALUDE_average_age_combined_l3468_346856

theorem average_age_combined (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℝ) (avg_age_parents : ℝ) :
  num_students = 40 →
  num_parents = 60 →
  avg_age_students = 10 →
  avg_age_parents = 40 →
  let total_individuals := num_students + num_parents
  let total_age := num_students * avg_age_students + num_parents * avg_age_parents
  (total_age / total_individuals : ℝ) = 28 := by
sorry

end NUMINAMATH_CALUDE_average_age_combined_l3468_346856


namespace NUMINAMATH_CALUDE_distribute_five_objects_l3468_346893

/-- The number of ways to distribute n distinguishable objects into 2 indistinguishable containers,
    such that neither container is empty -/
def distribute (n : ℕ) : ℕ :=
  (2^n - 2) / 2

/-- Theorem: There are 15 ways to distribute 5 distinguishable objects into 2 indistinguishable containers,
    such that neither container is empty -/
theorem distribute_five_objects : distribute 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_objects_l3468_346893


namespace NUMINAMATH_CALUDE_modulus_range_of_complex_l3468_346874

theorem modulus_range_of_complex (Z : ℂ) (a : ℝ) (h1 : 0 < a) (h2 : a < 2) 
  (h3 : Z.re = a) (h4 : Z.im = 1) : 1 < Complex.abs Z ∧ Complex.abs Z < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_range_of_complex_l3468_346874


namespace NUMINAMATH_CALUDE_chord_bisection_l3468_346845

/-- The ellipse defined by x²/16 + y²/8 = 1 -/
def Ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 8 = 1

/-- A point (x, y) lies on the line x + y - 3 = 0 -/
def Line (x y : ℝ) : Prop := x + y - 3 = 0

/-- The midpoint of two points -/
def Midpoint (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

theorem chord_bisection (x₁ y₁ x₂ y₂ : ℝ) :
  Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧ Midpoint x₁ y₁ x₂ y₂ 2 1 →
  Line x₁ y₁ ∧ Line x₂ y₂ := by
  sorry

end NUMINAMATH_CALUDE_chord_bisection_l3468_346845


namespace NUMINAMATH_CALUDE_isosceles_angle_B_l3468_346816

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the property of an isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Define the exterior angle of A
def exteriorAngleA (t : Triangle) : ℝ :=
  180 - t.A

-- Theorem statement
theorem isosceles_angle_B (t : Triangle) 
  (h_ext : exteriorAngleA t = 110) :
  isIsosceles t → t.B = 70 ∨ t.B = 55 ∨ t.B = 40 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_angle_B_l3468_346816


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l3468_346861

theorem sum_of_quadratic_solutions : 
  let f (x : ℝ) := x^2 - 4*x - 14 - (3*x + 16)
  let solutions := {x : ℝ | f x = 0}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l3468_346861


namespace NUMINAMATH_CALUDE_max_gcd_value_l3468_346840

def a (n : ℕ+) : ℕ := 121 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_value :
  (∃ (k : ℕ+), d k = 99) ∧ (∀ (n : ℕ+), d n ≤ 99) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_value_l3468_346840


namespace NUMINAMATH_CALUDE_rook_paths_bound_l3468_346826

def ChessboardPaths (n : ℕ) : ℕ := sorry

theorem rook_paths_bound (n : ℕ) :
  ChessboardPaths n ≤ 9^n ∧ ∀ k < 9, ∃ m : ℕ, ChessboardPaths m > k^m :=
by sorry

end NUMINAMATH_CALUDE_rook_paths_bound_l3468_346826


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3468_346827

theorem greatest_divisor_four_consecutive_integers :
  ∃ (d : ℕ), d = 12 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (m * (m + 1) * (m + 2) * (m + 3)))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3468_346827


namespace NUMINAMATH_CALUDE_y_value_l3468_346882

theorem y_value (x y : ℝ) (h1 : 2 * (x - y) = 12) (h2 : x + y = 14) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l3468_346882


namespace NUMINAMATH_CALUDE_regular_pentagon_angle_excess_prove_regular_pentagon_angle_excess_l3468_346857

theorem regular_pentagon_angle_excess : ℝ → Prop :=
  λ total_excess : ℝ =>
    -- Define a regular pentagon
    ∃ (interior_angle : ℝ),
      -- The sum of interior angles of a pentagon is (5-2)*180 = 540 degrees
      5 * interior_angle = 540 ∧
      -- The total excess over 90 degrees for all angles
      5 * (interior_angle - 90) = total_excess ∧
      -- The theorem to prove
      total_excess = 90

-- The proof of the theorem
theorem prove_regular_pentagon_angle_excess :
  ∃ total_excess : ℝ, regular_pentagon_angle_excess total_excess :=
by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_angle_excess_prove_regular_pentagon_angle_excess_l3468_346857


namespace NUMINAMATH_CALUDE_alarm_clock_noon_time_l3468_346888

/-- Represents the time difference between a correct clock and a slow clock -/
def timeDifference (slowRate : ℚ) (elapsedTime : ℚ) : ℚ :=
  slowRate * elapsedTime

/-- Calculates the time until noon for the slow clock -/
def timeUntilNoon (slowRate : ℚ) (elapsedTime : ℚ) : ℚ :=
  60 - (60 - timeDifference slowRate elapsedTime) % 60

theorem alarm_clock_noon_time (slowRate elapsedTime : ℚ) 
  (h1 : slowRate = 4 / 60) 
  (h2 : elapsedTime = 7 / 2) : 
  timeUntilNoon slowRate elapsedTime = 14 := by
sorry

#eval timeUntilNoon (4/60) (7/2)

end NUMINAMATH_CALUDE_alarm_clock_noon_time_l3468_346888


namespace NUMINAMATH_CALUDE_intersection_S_T_l3468_346829

-- Define the sets S and T
def S : Set ℝ := {x | x < -5 ∨ x > 5}
def T : Set ℝ := {x | -7 < x ∧ x < 3}

-- State the theorem
theorem intersection_S_T : S ∩ T = {x | -7 < x ∧ x < -5} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l3468_346829


namespace NUMINAMATH_CALUDE_jaces_remaining_money_l3468_346834

/-- Proves that Jace's remaining money after transactions is correct -/
theorem jaces_remaining_money
  (earnings : ℚ)
  (debt : ℚ)
  (neighbor_percentage : ℚ)
  (exchange_rate : ℚ)
  (h1 : earnings = 1500)
  (h2 : debt = 358)
  (h3 : neighbor_percentage = 1/4)
  (h4 : exchange_rate = 121/100) :
  earnings - debt - (earnings - debt) * neighbor_percentage = 8565/10 :=
by sorry


end NUMINAMATH_CALUDE_jaces_remaining_money_l3468_346834


namespace NUMINAMATH_CALUDE_box_of_balls_l3468_346822

theorem box_of_balls (x : ℕ) : 
  (25 - x = 30 - 25) → x = 20 := by sorry

end NUMINAMATH_CALUDE_box_of_balls_l3468_346822


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3468_346896

theorem inequality_equivalence (x y : ℝ) : y - x^2 < |x| ↔ y < x^2 + |x| := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3468_346896


namespace NUMINAMATH_CALUDE_all_transformed_points_in_S_l3468_346819

def S : Set ℂ := {z | -1 ≤ z.re ∧ z.re ≤ 1 ∧ -1 ≤ z.im ∧ z.im ≤ 1}

theorem all_transformed_points_in_S : ∀ z ∈ S, (1/2 + 1/2*I)*z ∈ S := by
  sorry

end NUMINAMATH_CALUDE_all_transformed_points_in_S_l3468_346819


namespace NUMINAMATH_CALUDE_total_selling_price_proof_l3468_346899

/-- Calculates the total selling price of toys given the number of toys sold,
    the cost price per toy, and the number of toys whose cost price equals the total gain. -/
def totalSellingPrice (numToysSold : ℕ) (costPricePerToy : ℕ) (numToysGain : ℕ) : ℕ :=
  numToysSold * costPricePerToy + numToysGain * costPricePerToy

/-- Proves that the total selling price of 18 toys is 16800,
    given a cost price of 800 per toy and a gain equal to the cost of 3 toys. -/
theorem total_selling_price_proof :
  totalSellingPrice 18 800 3 = 16800 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_proof_l3468_346899


namespace NUMINAMATH_CALUDE_gcd_13247_36874_l3468_346873

theorem gcd_13247_36874 : Nat.gcd 13247 36874 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_13247_36874_l3468_346873


namespace NUMINAMATH_CALUDE_sum_base8_equals_467_l3468_346839

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Adds two base-8 numbers --/
def addBase8 (a b : ℕ) : ℕ := base10ToBase8 (base8ToBase10 a + base8ToBase10 b)

theorem sum_base8_equals_467 :
  addBase8 (addBase8 236 157) 52 = 467 := by sorry

end NUMINAMATH_CALUDE_sum_base8_equals_467_l3468_346839


namespace NUMINAMATH_CALUDE_sandbag_weight_increase_l3468_346852

/-- Proves that the percentage increase in weight of a heavier filling material compared to sand is 40% given specific conditions. -/
theorem sandbag_weight_increase (capacity : ℝ) (fill_level : ℝ) (actual_weight : ℝ) : 
  capacity = 250 →
  fill_level = 0.8 →
  actual_weight = 280 →
  (actual_weight - fill_level * capacity) / (fill_level * capacity) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_sandbag_weight_increase_l3468_346852


namespace NUMINAMATH_CALUDE_linear_dependence_condition_l3468_346880

/-- Two 2D vectors are linearly dependent -/
def linearlyDependent (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), (a, b) ≠ (0, 0) ∧ a • v1 + b • v2 = (0, 0)

/-- The main theorem: vectors (2, 4) and (5, p) are linearly dependent iff p = 10 -/
theorem linear_dependence_condition (p : ℝ) :
  linearlyDependent (2, 4) (5, p) ↔ p = 10 := by
  sorry


end NUMINAMATH_CALUDE_linear_dependence_condition_l3468_346880


namespace NUMINAMATH_CALUDE_equation_true_iff_m_zero_l3468_346811

theorem equation_true_iff_m_zero (m n : ℝ) :
  21 * (m + n) + 21 = 21 * (-m + n) + 21 ↔ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_true_iff_m_zero_l3468_346811


namespace NUMINAMATH_CALUDE_jeans_price_ratio_l3468_346825

theorem jeans_price_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let sale_price := marked_price / 2
  let cost := (5 / 8) * sale_price
  cost / marked_price = 5 / 16 := by
sorry

end NUMINAMATH_CALUDE_jeans_price_ratio_l3468_346825


namespace NUMINAMATH_CALUDE_apples_in_baskets_l3468_346869

theorem apples_in_baskets (total_apples : ℕ) (num_baskets : ℕ) (apples_removed : ℕ) : 
  total_apples = 64 → num_baskets = 4 → apples_removed = 3 →
  (total_apples / num_baskets) - apples_removed = 13 :=
by
  sorry

#check apples_in_baskets

end NUMINAMATH_CALUDE_apples_in_baskets_l3468_346869


namespace NUMINAMATH_CALUDE_max_guaranteed_points_is_34_l3468_346859

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  num_teams : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- The specific tournament described in the problem -/
def tournament : FootballTournament :=
  { num_teams := 15
  , points_for_win := 3
  , points_for_draw := 1
  , points_for_loss := 0 }

/-- The maximum number of points that can be guaranteed for each of 6 teams -/
def max_guaranteed_points (t : FootballTournament) : Nat :=
  34

/-- Theorem stating that 34 is the maximum number of points that can be guaranteed for each of 6 teams -/
theorem max_guaranteed_points_is_34 :
  ∀ n : Nat, n > max_guaranteed_points tournament →
  ¬(∃ points : Fin tournament.num_teams → Nat,
    (∀ i j : Fin tournament.num_teams, i ≠ j →
      points i + points j ≤ tournament.points_for_win) ∧
    (∃ top_6 : Finset (Fin tournament.num_teams),
      top_6.card = 6 ∧ ∀ i ∈ top_6, points i ≥ n)) :=
by sorry

end NUMINAMATH_CALUDE_max_guaranteed_points_is_34_l3468_346859


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3468_346877

-- Define the distance traveled in the first and second hours
def distance_first_hour : ℝ := 98
def distance_second_hour : ℝ := 70

-- Define the total time
def total_time : ℝ := 2

-- Theorem statement
theorem average_speed_calculation :
  let total_distance := distance_first_hour + distance_second_hour
  (total_distance / total_time) = 84 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3468_346877


namespace NUMINAMATH_CALUDE_square_side_length_l3468_346830

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 1/4 → side^2 = area → side = 1/2 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l3468_346830


namespace NUMINAMATH_CALUDE_anthony_pencil_count_l3468_346812

/-- Anthony's initial pencil count -/
def initial_pencils : ℝ := 56.0

/-- Number of pencils Anthony gives away -/
def pencils_given : ℝ := 9.0

/-- Anthony's final pencil count -/
def final_pencils : ℝ := 47.0

theorem anthony_pencil_count : initial_pencils - pencils_given = final_pencils := by
  sorry

end NUMINAMATH_CALUDE_anthony_pencil_count_l3468_346812


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l3468_346828

theorem smallest_fraction_between (p q : ℕ+) : 
  (5 : ℚ) / 9 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (4 : ℚ) / 7 ∧ 
  (∀ p' q' : ℕ+, (5 : ℚ) / 9 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (4 : ℚ) / 7 → q ≤ q') →
  q - p = 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l3468_346828


namespace NUMINAMATH_CALUDE_gcd_228_1995_base3_11102_to_decimal_l3468_346865

-- Problem 1: GCD of 228 and 1995
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by sorry

-- Problem 2: Base 3 to decimal conversion
def base3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base3_11102_to_decimal :
  base3_to_decimal [1, 1, 1, 0, 2] = 119 := by sorry

end NUMINAMATH_CALUDE_gcd_228_1995_base3_11102_to_decimal_l3468_346865
