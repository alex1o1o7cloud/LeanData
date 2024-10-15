import Mathlib

namespace NUMINAMATH_GPT_evaluate_expression_l1396_139697

theorem evaluate_expression :
  (3 ^ (1 ^ (0 ^ 8)) + ( (3 ^ 1) ^ 0 ) ^ 8) = 4 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1396_139697


namespace NUMINAMATH_GPT_interval_solution_length_l1396_139698

theorem interval_solution_length (a b : ℝ) (h : (b - a) / 3 = 8) : b - a = 24 := by
  sorry

end NUMINAMATH_GPT_interval_solution_length_l1396_139698


namespace NUMINAMATH_GPT_ticket_sales_amount_theater_collected_50_dollars_l1396_139638

variable (num_people total_people : ℕ) (cost_adult_entry cost_child_entry : ℕ) (num_children : ℕ)
variable (total_collected : ℕ)

theorem ticket_sales_amount
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : num_people = total_people - num_children)
  : total_collected = (num_people * cost_adult_entry + num_children * cost_child_entry) := sorry

theorem theater_collected_50_dollars 
  (h1 : cost_adult_entry = 8)
  (h2 : cost_child_entry = 1)
  (h3 : total_people = 22)
  (h4 : num_children = 18)
  (h5 : total_collected = 50)
  : total_collected = 50 := sorry

end NUMINAMATH_GPT_ticket_sales_amount_theater_collected_50_dollars_l1396_139638


namespace NUMINAMATH_GPT_poly_roots_equivalence_l1396_139668

noncomputable def poly (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem poly_roots_equivalence (a b c d : ℝ) 
    (h1 : poly a b c d 4 = 102) 
    (h2 : poly a b c d 3 = 102) 
    (h3 : poly a b c d (-3) = 102) 
    (h4 : poly a b c d (-4) = 102) : 
    {x : ℝ | poly a b c d x = 246} = {0, 5, -5} := 
by 
    sorry

end NUMINAMATH_GPT_poly_roots_equivalence_l1396_139668


namespace NUMINAMATH_GPT_multiply_powers_l1396_139669

theorem multiply_powers (x : ℝ) : x^3 * x^3 = x^6 :=
by sorry

end NUMINAMATH_GPT_multiply_powers_l1396_139669


namespace NUMINAMATH_GPT_inequality_proof_l1396_139615

theorem inequality_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  a^a * b^b + a^b * b^a ≤ 1 :=
  sorry

end NUMINAMATH_GPT_inequality_proof_l1396_139615


namespace NUMINAMATH_GPT_tom_received_20_percent_bonus_l1396_139685

-- Define the initial conditions
def tom_spent : ℤ := 250
def gems_per_dollar : ℤ := 100
def total_gems_received : ℤ := 30000

-- Calculate the number of gems received without the bonus
def gems_without_bonus : ℤ := tom_spent * gems_per_dollar
def bonus_gems : ℤ := total_gems_received - gems_without_bonus

-- Calculate the percentage of the bonus
def bonus_percentage : ℚ := (bonus_gems : ℚ) / gems_without_bonus * 100

-- State the theorem
theorem tom_received_20_percent_bonus : bonus_percentage = 20 := by
  sorry

end NUMINAMATH_GPT_tom_received_20_percent_bonus_l1396_139685


namespace NUMINAMATH_GPT_problem_solution_l1396_139673

theorem problem_solution (a b c d : ℕ) (h : 342 * (a * b * c * d + a * b + a * d + c * d + 1) = 379 * (b * c * d + b + d)) :
  (a * 10^3 + b * 10^2 + c * 10 + d) = 1949 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1396_139673


namespace NUMINAMATH_GPT_inequality_proof_l1396_139679

variable (x y z : ℝ)

theorem inequality_proof (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  x * (1 - 2 * x) * (1 - 3 * x) + y * (1 - 2 * y) * (1 - 3 * y) + z * (1 - 2 * z) * (1 - 3 * z) ≥ 0 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1396_139679


namespace NUMINAMATH_GPT_min_sqrt_diff_l1396_139600

theorem min_sqrt_diff (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ x y : ℕ, x = (p - 1) / 2 ∧ y = (p + 1) / 2 ∧ x ≤ y ∧
    ∀ a b : ℕ, (a ≤ b) → (Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b ≥ 0) → 
      (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y) ≤ (Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) := 
by 
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_min_sqrt_diff_l1396_139600


namespace NUMINAMATH_GPT_rebecca_tent_stakes_l1396_139675

variables (T D W : ℕ)

-- Conditions
def drink_mix_eq : Prop := D = 3 * T
def water_eq : Prop := W = T + 2
def total_items_eq : Prop := T + D + W = 22

-- Problem statement
theorem rebecca_tent_stakes 
  (h1 : drink_mix_eq T D)
  (h2 : water_eq T W)
  (h3 : total_items_eq T D W) : 
  T = 4 := 
sorry

end NUMINAMATH_GPT_rebecca_tent_stakes_l1396_139675


namespace NUMINAMATH_GPT_find_a_l1396_139694

def A := { x : ℝ | x^2 + 4 * x = 0 }
def B (a : ℝ) := { x : ℝ | x^2 + 2 * (a + 1) * x + (a^2 - 1) = 0 }

theorem find_a (a : ℝ) :
  (∀ x : ℝ, x ∈ (A ∩ B a) ↔ x ∈ B a) → (a = 1 ∨ a ≤ -1) :=
by 
  sorry

end NUMINAMATH_GPT_find_a_l1396_139694


namespace NUMINAMATH_GPT_shopkeeper_profit_percentage_l1396_139624

theorem shopkeeper_profit_percentage 
  (cost_price : ℝ := 100) 
  (loss_due_to_theft_percent : ℝ := 30) 
  (overall_loss_percent : ℝ := 23) 
  (remaining_goods_value : ℝ := 70) 
  (overall_loss_value : ℝ := 23) 
  (selling_price : ℝ := 77) 
  (profit_percentage : ℝ) 
  (h1 : remaining_goods_value = cost_price * (1 - loss_due_to_theft_percent / 100)) 
  (h2 : overall_loss_value = cost_price * (overall_loss_percent / 100)) 
  (h3 : selling_price = cost_price - overall_loss_value) 
  (h4 : remaining_goods_value + remaining_goods_value * profit_percentage / 100 = selling_price) :
  profit_percentage = 10 := 
by 
  sorry

end NUMINAMATH_GPT_shopkeeper_profit_percentage_l1396_139624


namespace NUMINAMATH_GPT_triangle_geometry_l1396_139699

theorem triangle_geometry 
  (A : ℝ × ℝ) 
  (hA : A = (5,1))
  (median_CM : ∀ x y : ℝ, 2 * x - y - 5 = 0)
  (altitude_BH : ∀ x y : ℝ, x - 2 * y - 5 = 0):
  (∀ x y : ℝ, 2 * x + y - 11 = 0) ∧
  (4, 3) ∈ {(x, y) | 2 * x + y = 11 ∧ 2 * x - y = 5} :=
by
  sorry

end NUMINAMATH_GPT_triangle_geometry_l1396_139699


namespace NUMINAMATH_GPT_minimal_sum_of_squares_of_roots_l1396_139676

open Real

theorem minimal_sum_of_squares_of_roots :
  ∀ a : ℝ,
  (let x1 := 3*a + 1;
   let x2 := 2*a^2 - 3*a - 2;
   (a^2 + 18*a + 9) ≥ 0 →
   (x1^2 - 2*x2) = (5*a^2 + 12*a + 5) →
   a = -9 + 6*sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_minimal_sum_of_squares_of_roots_l1396_139676


namespace NUMINAMATH_GPT_find_tony_age_l1396_139678

variable (y : ℕ)
variable (d : ℕ)

def Tony_day_hours : ℕ := 3
def Tony_hourly_rate (age : ℕ) : ℚ := 0.75 * age
def Tony_days_worked : ℕ := 60
def Tony_total_earnings : ℚ := 945

noncomputable def earnings_before_birthday (age : ℕ) (days : ℕ) : ℚ :=
  Tony_hourly_rate age * Tony_day_hours * days

noncomputable def earnings_after_birthday (age : ℕ) (days : ℕ) : ℚ :=
  Tony_hourly_rate (age + 1) * Tony_day_hours * days

noncomputable def total_earnings (age : ℕ) (days_before : ℕ) : ℚ :=
  (earnings_before_birthday age days_before) +
  (earnings_after_birthday age (Tony_days_worked - days_before))

theorem find_tony_age: ∃ y d : ℕ, total_earnings y d = Tony_total_earnings ∧ y = 6 := by
  sorry

end NUMINAMATH_GPT_find_tony_age_l1396_139678


namespace NUMINAMATH_GPT_martha_saves_half_daily_allowance_l1396_139630

theorem martha_saves_half_daily_allowance {f : ℚ} (h₁ : 12 > 0) (h₂ : (6 : ℚ) * 12 * f + (3 : ℚ) = 39) : f = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_martha_saves_half_daily_allowance_l1396_139630


namespace NUMINAMATH_GPT_area_of_sector_l1396_139603

noncomputable def circleAreaAboveXAxisAndRightOfLine : ℝ :=
  let radius := 10
  let area_of_circle := Real.pi * radius^2
  area_of_circle / 4

theorem area_of_sector :
  circleAreaAboveXAxisAndRightOfLine = 25 * Real.pi := sorry

end NUMINAMATH_GPT_area_of_sector_l1396_139603


namespace NUMINAMATH_GPT_taxi_ride_cost_l1396_139609

-- Define the base fare
def base_fare : ℝ := 2.00

-- Define the cost per mile
def cost_per_mile : ℝ := 0.30

-- Define the distance traveled
def distance : ℝ := 8.00

-- Define the total cost function
def total_cost (base : ℝ) (per_mile : ℝ) (miles : ℝ) : ℝ :=
  base + (per_mile * miles)

-- The statement to prove: the total cost of an 8-mile taxi ride
theorem taxi_ride_cost : total_cost base_fare cost_per_mile distance = 4.40 :=
by
sorry

end NUMINAMATH_GPT_taxi_ride_cost_l1396_139609


namespace NUMINAMATH_GPT_length_of_P1P2_segment_l1396_139622

theorem length_of_P1P2_segment (x : ℝ) (h₀ : 0 < x ∧ x < π / 2) (h₁ : 6 * Real.cos x = 9 * Real.tan x) :
  Real.sin x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_length_of_P1P2_segment_l1396_139622


namespace NUMINAMATH_GPT_man_speed_l1396_139657

theorem man_speed {m l: ℝ} (TrainLength : ℝ := 385) (TrainSpeedKmH : ℝ := 60)
  (PassTimeSeconds : ℝ := 21) (RelativeSpeed : ℝ) (ManSpeedKmH : ℝ) 
  (ConversionFactor : ℝ := 3.6) (expected_speed : ℝ := 5.99) : 
  RelativeSpeed = TrainSpeedKmH/ConversionFactor + m/ConversionFactor ∧ 
  TrainLength = RelativeSpeed * PassTimeSeconds →
  abs (m*ConversionFactor - expected_speed) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_man_speed_l1396_139657


namespace NUMINAMATH_GPT_inequality_solution_l1396_139634

theorem inequality_solution (x : ℝ) (h : x ≠ -5) : 
  (x^2 - 25) / (x + 5) < 0 ↔ x ∈ Set.union (Set.Iio (-5)) (Set.Ioo (-5) 5) := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1396_139634


namespace NUMINAMATH_GPT_train_still_there_when_susan_arrives_l1396_139632

-- Define the conditions and primary question
def time_between_1_and_2 (t : ℝ) : Prop := 0 ≤ t ∧ t ≤ 60

def train_arrival := {t : ℝ // time_between_1_and_2 t}
def susan_arrival := {t : ℝ // time_between_1_and_2 t}

def train_present (train : train_arrival) (susan : susan_arrival) : Prop :=
  susan.val ≥ train.val ∧ susan.val ≤ (train.val + 30)

-- Define the probability calculation
noncomputable def probability_train_present : ℝ :=
  (30 * 30 + (30 * (60 - 30) * 2) / 2) / (60 * 60)

theorem train_still_there_when_susan_arrives :
  probability_train_present = 1 / 2 :=
sorry

end NUMINAMATH_GPT_train_still_there_when_susan_arrives_l1396_139632


namespace NUMINAMATH_GPT_toms_age_is_16_l1396_139656

variable (J T : ℕ) -- John's current age is J and Tom's current age is T

-- Condition 1: John was thrice as old as Tom 6 years ago
axiom h1 : J - 6 = 3 * (T - 6)

-- Condition 2: John will be 2 times as old as Tom in 4 years
axiom h2 : J + 4 = 2 * (T + 4)

-- Proving Tom's current age is 16
theorem toms_age_is_16 : T = 16 := by
  sorry

end NUMINAMATH_GPT_toms_age_is_16_l1396_139656


namespace NUMINAMATH_GPT_solve_for_x_l1396_139693

theorem solve_for_x (x y : ℚ) :
  (x + 1) / (x - 2) = (y^2 + 4*y + 1) / (y^2 + 4*y - 3) →
  x = -(3*y^2 + 12*y - 1) / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1396_139693


namespace NUMINAMATH_GPT_car_distance_l1396_139677

-- Define the conditions
def speed := 162  -- speed of the car in km/h
def time := 5     -- time taken in hours

-- Define the distance calculation
def distance (s : ℕ) (t : ℕ) : ℕ := s * t

-- State the theorem
theorem car_distance : distance speed time = 810 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_car_distance_l1396_139677


namespace NUMINAMATH_GPT_scientific_notation_of_0_00065_l1396_139659

/-- 
Prove that the decimal representation of a number 0.00065 can be expressed in scientific notation 
as 6.5 * 10^(-4)
-/
theorem scientific_notation_of_0_00065 : 0.00065 = 6.5 * 10^(-4) := 
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_of_0_00065_l1396_139659


namespace NUMINAMATH_GPT_total_marbles_l1396_139681

theorem total_marbles (bowl2_capacity : ℕ) (h₁ : bowl2_capacity = 600)
    (h₂ : 3 / 4 * bowl2_capacity = 450) : 600 + (3 / 4 * 600) = 1050 := by
  sorry

end NUMINAMATH_GPT_total_marbles_l1396_139681


namespace NUMINAMATH_GPT_employed_females_percentage_l1396_139649

theorem employed_females_percentage (total_population : ℝ) (total_employed_percentage : ℝ) (employed_males_percentage : ℝ) :
  total_employed_percentage = 0.7 →
  employed_males_percentage = 0.21 →
  total_population > 0 →
  (total_employed_percentage - employed_males_percentage) / total_employed_percentage * 100 = 70 :=
by
  intros h1 h2 h3
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_employed_females_percentage_l1396_139649


namespace NUMINAMATH_GPT_balloons_given_by_mom_l1396_139639

def num_balloons_initial : ℕ := 26
def num_balloons_total : ℕ := 60

theorem balloons_given_by_mom :
  (num_balloons_total - num_balloons_initial) = 34 := 
by
  sorry

end NUMINAMATH_GPT_balloons_given_by_mom_l1396_139639


namespace NUMINAMATH_GPT_initially_calculated_average_height_l1396_139601

theorem initially_calculated_average_height
    (A : ℕ)
    (initial_total_height : ℕ)
    (real_total_height : ℕ)
    (height_error : ℕ := 60)
    (num_boys : ℕ := 35)
    (actual_average_height : ℕ := 183)
    (initial_total_height_eq : initial_total_height = num_boys * A)
    (real_total_height_eq : real_total_height = num_boys * actual_average_height)
    (height_discrepancy : initial_total_height = real_total_height + height_error) :
    A = 181 :=
by
  sorry

end NUMINAMATH_GPT_initially_calculated_average_height_l1396_139601


namespace NUMINAMATH_GPT_find_k_value_l1396_139607

theorem find_k_value (Z K : ℤ) (h1 : 1000 < Z) (h2 : Z < 8000) (h3 : K > 2) (h4 : Z = K^3)
  (h5 : ∃ n : ℤ, Z = n^6) : K = 16 :=
sorry

end NUMINAMATH_GPT_find_k_value_l1396_139607


namespace NUMINAMATH_GPT_sum_multiple_of_three_l1396_139686

theorem sum_multiple_of_three (a b : ℤ) (h₁ : ∃ m, a = 6 * m) (h₂ : ∃ n, b = 9 * n) : ∃ k, (a + b) = 3 * k :=
by
  sorry

end NUMINAMATH_GPT_sum_multiple_of_three_l1396_139686


namespace NUMINAMATH_GPT_seed_total_after_trading_l1396_139692

theorem seed_total_after_trading :
  ∀ (Bom Gwi Yeon Eun : ℕ),
  Yeon = 3 * Gwi →
  Gwi = Bom + 40 →
  Eun = 2 * Gwi →
  Bom = 300 →
  Yeon_gives = 20 * Yeon / 100 →
  Bom_gives = 50 →
  let Yeon_after := Yeon - Yeon_gives
  let Gwi_after := Gwi + Yeon_gives
  let Bom_after := Bom - Bom_gives
  let Eun_after := Eun + Bom_gives
  Bom_after + Gwi_after + Yeon_after + Eun_after = 2340 :=
by
  intros Bom Gwi Yeon Eun hYeon hGwi hEun hBom hYeonGives hBomGives Yeon_after Gwi_after Bom_after Eun_after
  sorry

end NUMINAMATH_GPT_seed_total_after_trading_l1396_139692


namespace NUMINAMATH_GPT_line_intercepts_and_slope_l1396_139610

theorem line_intercepts_and_slope :
  ∀ (x y : ℝ), (4 * x - 5 * y - 20 = 0) → 
  ∃ (x_intercept : ℝ) (y_intercept : ℝ) (slope : ℝ), 
    x_intercept = 5 ∧ y_intercept = -4 ∧ slope = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_line_intercepts_and_slope_l1396_139610


namespace NUMINAMATH_GPT_tan_theta_solution_l1396_139690

theorem tan_theta_solution (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < 15) 
  (h_tan_eq : Real.tan θ + Real.tan (2 * θ) + Real.tan (4 * θ) = 0) :
  Real.tan θ = 1 / Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_tan_theta_solution_l1396_139690


namespace NUMINAMATH_GPT_exists_trinomial_with_exponents_three_l1396_139625

theorem exists_trinomial_with_exponents_three (x y : ℝ) :
  ∃ (a b c : ℝ) (t1 t2 t3 : ℕ × ℕ), 
  t1.1 + t1.2 = 3 ∧ t2.1 + t2.2 = 3 ∧ t3.1 + t3.2 = 3 ∧
  (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
  (a * x ^ t1.1 * y ^ t1.2 + b * x ^ t2.1 * y ^ t2.2 + c * x ^ t3.1 * y ^ t3.2 ≠ 0) := sorry

end NUMINAMATH_GPT_exists_trinomial_with_exponents_three_l1396_139625


namespace NUMINAMATH_GPT_minimum_value_xyz_l1396_139651

theorem minimum_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) : 
  ∃ m : ℝ, m = 16 ∧ ∀ w, w = (x + y) / (x * y * z) → w ≥ m :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_xyz_l1396_139651


namespace NUMINAMATH_GPT_prime_cube_plus_five_implies_prime_l1396_139689

theorem prime_cube_plus_five_implies_prime (p : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime (p^3 + 5)) : p^5 - 7 = 25 := 
by
  sorry

end NUMINAMATH_GPT_prime_cube_plus_five_implies_prime_l1396_139689


namespace NUMINAMATH_GPT_melissa_total_time_l1396_139628

-- Definitions based on the conditions in the problem
def time_replace_buckle : Nat := 5
def time_even_heel : Nat := 10
def time_fix_straps : Nat := 7
def time_reattach_soles : Nat := 12
def pairs_of_shoes : Nat := 8

-- Translation of the mathematically equivalent proof problem
theorem melissa_total_time : 
  (time_replace_buckle + time_even_heel + time_fix_straps + time_reattach_soles) * 16 = 544 :=
by
  sorry

end NUMINAMATH_GPT_melissa_total_time_l1396_139628


namespace NUMINAMATH_GPT_sum_of_constants_eq_zero_l1396_139618

theorem sum_of_constants_eq_zero (A B C D E : ℝ) :
  (∀ (x : ℝ), (x + 1) / ((x + 2) * (x + 3) * (x + 4) * (x + 5) * (x + 6)) =
              A / (x + 2) + B / (x + 3) + C / (x + 4) + D / (x + 5) + E / (x + 6)) →
  A + B + C + D + E = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_constants_eq_zero_l1396_139618


namespace NUMINAMATH_GPT_tangent_line_sin_at_pi_l1396_139636

theorem tangent_line_sin_at_pi :
  ∀ (f : ℝ → ℝ), 
    (∀ x, f x = Real.sin x) → ∀ x y, (x, y) = (Real.pi, 0) → 
    ∃ (m : ℝ) (b : ℝ), (∀ x, y = m * x + b) ∧ (m = -1) ∧ (b = Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_sin_at_pi_l1396_139636


namespace NUMINAMATH_GPT_average_rate_of_interest_l1396_139645

/-- Given:
    1. A woman has a total of $7500 invested,
    2. Part of the investment is at 5% interest,
    3. The remainder of the investment is at 7% interest,
    4. The annual returns from both investments are equal,
    Prove:
    The average rate of interest realized on her total investment is 5.8%.
-/
theorem average_rate_of_interest
  (total_investment : ℝ) (interest_5_percent : ℝ) (interest_7_percent : ℝ)
  (annual_return_equal : 0.05 * (total_investment - interest_7_percent) = 0.07 * interest_7_percent)
  (total_investment_eq : total_investment = 7500) : 
  (interest_5_percent / total_investment) = 0.058 :=
by
  -- conditions given
  have h1 : total_investment = 7500 := total_investment_eq
  have h2 : 0.05 * (total_investment - interest_7_percent) = 0.07 * interest_7_percent := annual_return_equal

  -- final step, sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_average_rate_of_interest_l1396_139645


namespace NUMINAMATH_GPT_series_sum_eq_l1396_139660

theorem series_sum_eq : 
  (∑' n, (4 * n + 3) / ((4 * n - 2) ^ 2 * (4 * n + 2) ^ 2)) = 1 / 128 := by
sorry

end NUMINAMATH_GPT_series_sum_eq_l1396_139660


namespace NUMINAMATH_GPT_mark_spending_l1396_139617

theorem mark_spending (initial_money : ℕ) (first_store_half : ℕ) (first_store_additional : ℕ) 
                      (second_store_third : ℕ) (remaining_money : ℕ) (total_spent : ℕ) : 
  initial_money = 180 ∧ 
  first_store_half = 90 ∧ 
  first_store_additional = 14 ∧ 
  total_spent = first_store_half + first_store_additional ∧
  remaining_money = initial_money - total_spent ∧
  second_store_third = 60 ∧ 
  remaining_money - second_store_third = 16 ∧ 
  initial_money - (total_spent + second_store_third + 16) = 0 → 
  remaining_money - second_store_third = 16 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mark_spending_l1396_139617


namespace NUMINAMATH_GPT_maxwell_walking_speed_l1396_139613

-- Define Maxwell's walking speed
def Maxwell_speed (v : ℕ) : Prop :=
  ∀ t1 t2 : ℕ, t1 = 10 → t2 = 9 →
  ∀ d1 d2 : ℕ, d1 = 10 * v → d2 = 6 * t2 →
  ∀ d_total : ℕ, d_total = 94 →
  d1 + d2 = d_total

theorem maxwell_walking_speed : Maxwell_speed 4 :=
by
  sorry

end NUMINAMATH_GPT_maxwell_walking_speed_l1396_139613


namespace NUMINAMATH_GPT_schoolchildren_chocolate_l1396_139654

theorem schoolchildren_chocolate (m d : ℕ) 
  (h1 : 7 * d + 2 * m > 36)
  (h2 : 8 * d + 4 * m < 48) :
  m = 1 ∧ d = 5 :=
by
  sorry

end NUMINAMATH_GPT_schoolchildren_chocolate_l1396_139654


namespace NUMINAMATH_GPT_average_salary_rest_l1396_139667

noncomputable def average_salary_of_the_rest : ℕ := 6000

theorem average_salary_rest 
  (N : ℕ) 
  (A : ℕ)
  (T : ℕ)
  (A_T : ℕ)
  (Nr : ℕ)
  (Ar : ℕ)
  (H1 : N = 42)
  (H2 : A = 8000)
  (H3 : T = 7)
  (H4 : A_T = 18000)
  (H5 : Nr = N - T)
  (H6 : Nr = 42 - 7)
  (H7 : Ar = 6000)
  (H8 : 42 * 8000 = (Nr * Ar) + (T * 18000))
  : Ar = average_salary_of_the_rest :=
by
  sorry

end NUMINAMATH_GPT_average_salary_rest_l1396_139667


namespace NUMINAMATH_GPT_snail_stops_at_25_26_l1396_139641

def grid_width : ℕ := 300
def grid_height : ℕ := 50

def initial_position : ℕ × ℕ := (1, 1)

def snail_moves_in_spiral (w h : ℕ) (initial : ℕ × ℕ) : ℕ × ℕ := (25, 26)

theorem snail_stops_at_25_26 :
  snail_moves_in_spiral grid_width grid_height initial_position = (25, 26) :=
sorry

end NUMINAMATH_GPT_snail_stops_at_25_26_l1396_139641


namespace NUMINAMATH_GPT_find_smaller_number_l1396_139619

def smaller_number (x y : ℕ) : ℕ :=
  if x < y then x else y

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 64) (h2 : a = b + 12) : smaller_number a b = 26 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l1396_139619


namespace NUMINAMATH_GPT_population_proof_l1396_139696

def population (tosses : ℕ) (values : ℕ) : Prop :=
  (tosses = 7768) ∧ (values = 6)

theorem population_proof : 
  population 7768 6 :=
by
  unfold population
  exact And.intro rfl rfl

end NUMINAMATH_GPT_population_proof_l1396_139696


namespace NUMINAMATH_GPT_jason_nickels_is_52_l1396_139664

theorem jason_nickels_is_52 (n q : ℕ) (h1 : 5 * n + 10 * q = 680) (h2 : q = n - 10) : n = 52 :=
sorry

end NUMINAMATH_GPT_jason_nickels_is_52_l1396_139664


namespace NUMINAMATH_GPT_cos_product_equals_one_eighth_l1396_139646

noncomputable def cos_pi_over_9 := Real.cos (Real.pi / 9)
noncomputable def cos_2pi_over_9 := Real.cos (2 * Real.pi / 9)
noncomputable def cos_4pi_over_9 := Real.cos (4 * Real.pi / 9)

theorem cos_product_equals_one_eighth :
  cos_pi_over_9 * cos_2pi_over_9 * cos_4pi_over_9 = 1 / 8 := 
sorry

end NUMINAMATH_GPT_cos_product_equals_one_eighth_l1396_139646


namespace NUMINAMATH_GPT_find_angle_B_in_right_triangle_l1396_139602

theorem find_angle_B_in_right_triangle (A B C : ℝ) (hC : C = 90) (hA : A = 35) :
  B = 55 :=
by
  -- Assuming A, B, and C represent the three angles of a triangle ABC
  -- where C = 90 degrees and A = 35 degrees, we need to prove B = 55 degrees.
  sorry

end NUMINAMATH_GPT_find_angle_B_in_right_triangle_l1396_139602


namespace NUMINAMATH_GPT_find_c_l1396_139633

/-- Seven unit squares are arranged in a row in the coordinate plane, 
with the lower left corner of the first square at the origin. 
A line extending from (c,0) to (4,4) divides the entire region 
into two regions of equal area. What is the value of c?
-/
theorem find_c (c : ℝ) (h : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 7 ∧ y = (4 / (4 - c)) * (x - c)) : c = 2.25 :=
sorry

end NUMINAMATH_GPT_find_c_l1396_139633


namespace NUMINAMATH_GPT_time_to_cross_pole_correct_l1396_139655

noncomputable def speed_kmph : ℝ := 160 -- Speed of the train in kmph
noncomputable def length_meters : ℝ := 800.064 -- Length of the train in meters

noncomputable def conversion_factor : ℝ := 1000 / 3600 -- Conversion factor from kmph to m/s
noncomputable def speed_mps : ℝ := speed_kmph * conversion_factor -- Speed of the train in m/s

noncomputable def time_to_cross_pole : ℝ := length_meters / speed_mps -- Time to cross the pole

theorem time_to_cross_pole_correct :
  time_to_cross_pole = 800.064 / (160 * (1000 / 3600)) :=
sorry

end NUMINAMATH_GPT_time_to_cross_pole_correct_l1396_139655


namespace NUMINAMATH_GPT_number_of_trumpet_players_l1396_139661

def number_of_people_in_orchestra := 21
def number_of_people_known := 1 -- Sebastian
                             + 4 -- Trombone players
                             + 1 -- French horn player
                             + 3 -- Violinists
                             + 1 -- Cellist
                             + 1 -- Contrabassist
                             + 3 -- Clarinet players
                             + 4 -- Flute players
                             + 1 -- Maestro

theorem number_of_trumpet_players : 
  number_of_people_in_orchestra = number_of_people_known + 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_trumpet_players_l1396_139661


namespace NUMINAMATH_GPT_fruit_problem_l1396_139666

def number_of_pears (A : ℤ) : ℤ := (3 * A) / 5
def number_of_apples (B : ℤ) : ℤ := (3 * B) / 7

theorem fruit_problem
  (A B : ℤ)
  (h1 : A + B = 82)
  (h2 : abs (A - B) < 10)
  (x : ℤ := (2 * A) / 5)
  (y : ℤ := (4 * B) / 7) :
  number_of_pears A = 24 ∧ number_of_apples B = 18 :=
by
  sorry

end NUMINAMATH_GPT_fruit_problem_l1396_139666


namespace NUMINAMATH_GPT_sum_of_vertices_l1396_139643

theorem sum_of_vertices (num_triangle num_hexagon : ℕ) (vertices_triangle vertices_hexagon : ℕ) :
  num_triangle = 1 → vertices_triangle = 3 →
  num_hexagon = 3 → vertices_hexagon = 6 →
  num_triangle * vertices_triangle + num_hexagon * vertices_hexagon = 21 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sum_of_vertices_l1396_139643


namespace NUMINAMATH_GPT_number_of_paths_l1396_139606

theorem number_of_paths (n m : ℕ) (h : m ≤ n) : 
  ∃ paths : ℕ, paths = Nat.choose n m := 
sorry

end NUMINAMATH_GPT_number_of_paths_l1396_139606


namespace NUMINAMATH_GPT_symmetric_point_min_value_l1396_139691

theorem symmetric_point_min_value (a b : ℝ) 
  (h1 : a > 0 ∧ b > 0) 
  (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ - 2 = 0 ∧ 2 * x₀ + y₀ + 3 = 0 ∧ 
        a + b = x₀ + y₀ ∧ ∃ k, k = (y₀ - b) / (x₀ - a) ∧ y₀ = k * x₀ + 2 - k * (a + k * b))
   : ∃ α β, a = β / α ∧  b = 2 * β / α ∧ (1 / a + 8 / b) = 25 / 9 :=
sorry

end NUMINAMATH_GPT_symmetric_point_min_value_l1396_139691


namespace NUMINAMATH_GPT_sum_of_first_nine_terms_l1396_139663

noncomputable def arithmetic_sequence_sum (a₁ d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def a_n (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem sum_of_first_nine_terms (a₁ d : ℕ) (h : a_n a₁ d 2 + a_n a₁ d 6 + a_n a₁ d 7 = 18) :
  arithmetic_sequence_sum a₁ d 9 = 54 :=
sorry

end NUMINAMATH_GPT_sum_of_first_nine_terms_l1396_139663


namespace NUMINAMATH_GPT_chord_length_eq_l1396_139642

def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

theorem chord_length_eq : 
  ∀ (x y : ℝ), 
  (line_eq x y) ∧ (circle_eq x y) → 
  ∃ l, l = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_chord_length_eq_l1396_139642


namespace NUMINAMATH_GPT_avg_weight_ab_l1396_139640

theorem avg_weight_ab (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 30) 
  (h2 : (B + C) / 2 = 28) 
  (h3 : B = 16) : 
  (A + B) / 2 = 25 := 
by 
  sorry

end NUMINAMATH_GPT_avg_weight_ab_l1396_139640


namespace NUMINAMATH_GPT_jim_profit_percentage_l1396_139620

theorem jim_profit_percentage (S C : ℝ) (H1 : S = 670) (H2 : C = 536) :
  ((S - C) / C) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_jim_profit_percentage_l1396_139620


namespace NUMINAMATH_GPT_complex_poly_root_exists_l1396_139688

noncomputable def polynomial_has_complex_root (P : Polynomial ℂ) : Prop :=
  ∃ z : ℂ, P.eval z = 0

theorem complex_poly_root_exists (P : Polynomial ℂ) : polynomial_has_complex_root P :=
sorry

end NUMINAMATH_GPT_complex_poly_root_exists_l1396_139688


namespace NUMINAMATH_GPT_identical_sets_l1396_139653

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x^2 + 1}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
def C : Set (ℝ × ℝ) := {(x, y) : ℝ × ℝ | y = x^2 + 1}
def D : Set ℝ := {y : ℝ | 1 ≤ y}

theorem identical_sets : B = D :=
by
  sorry

end NUMINAMATH_GPT_identical_sets_l1396_139653


namespace NUMINAMATH_GPT_trigonometric_identity_l1396_139658

theorem trigonometric_identity :
  4 * Real.cos (10 * (Real.pi / 180)) - Real.tan (80 * (Real.pi / 180)) = -Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1396_139658


namespace NUMINAMATH_GPT_line_equation_l1396_139682

theorem line_equation (A : (ℝ × ℝ)) (hA_x : A.1 = 2) (hA_y : A.2 = 0)
  (h_intercept : ∀ B : (ℝ × ℝ), B.1 = 0 → 2 * B.1 + B.2 + 2 = 0 → B = (0, -2)) :
  ∃ (l : ℝ × ℝ → Prop), (l A ∧ l (0, -2)) ∧ 
    (∀ x y : ℝ, l (x, y) ↔ x - y - 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l1396_139682


namespace NUMINAMATH_GPT_arithmetic_sqrt_of_nine_l1396_139629

-- Define the arithmetic square root function which only considers non-negative values
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  if hx : x ≥ 0 then Real.sqrt x else 0

-- The theorem to prove: The arithmetic square root of 9 is 3.
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sqrt_of_nine_l1396_139629


namespace NUMINAMATH_GPT_find_a_l1396_139637

def point_of_tangency (x0 y0 a : ℝ) : Prop :=
  (x0 - y0 - 1 = 0) ∧ (y0 = a * x0^2) ∧ (2 * a * x0 = 1)

theorem find_a (x0 y0 a : ℝ) (h : point_of_tangency x0 y0 a) : a = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1396_139637


namespace NUMINAMATH_GPT_algebraic_expression_value_l1396_139662

noncomputable def a := Real.sqrt 2 + 1
noncomputable def b := Real.sqrt 2 - 1

theorem algebraic_expression_value : (a^2 - 2 * a * b + b^2) / (a^2 - b^2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1396_139662


namespace NUMINAMATH_GPT_quotient_when_divided_by_5_l1396_139605

theorem quotient_when_divided_by_5 (N : ℤ) (k : ℤ) (Q : ℤ) 
  (h1 : N = 5 * Q) 
  (h2 : N % 4 = 2) : 
  Q = 2 := 
sorry

end NUMINAMATH_GPT_quotient_when_divided_by_5_l1396_139605


namespace NUMINAMATH_GPT_sqrt_sum_of_four_terms_of_4_pow_4_l1396_139665

-- Proof Statement
theorem sqrt_sum_of_four_terms_of_4_pow_4 : 
  Real.sqrt (4 ^ 4 + 4 ^ 4 + 4 ^ 4 + 4 ^ 4) = 32 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_sum_of_four_terms_of_4_pow_4_l1396_139665


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1396_139672

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) 
(h : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) 
(h_a : ∀ n, a n = a 1 + (n - 1) * d) : a 2 + a 10 = 120 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1396_139672


namespace NUMINAMATH_GPT_parametric_to_standard_form_l1396_139644

theorem parametric_to_standard_form (t : ℝ) (x y : ℝ)
    (param_eq1 : x = 1 + t)
    (param_eq2 : y = -1 + t) :
    x - y - 2 = 0 :=
sorry

end NUMINAMATH_GPT_parametric_to_standard_form_l1396_139644


namespace NUMINAMATH_GPT_cylinder_lateral_surface_area_l1396_139684

theorem cylinder_lateral_surface_area :
  let side := 20
  let radius := side / 2
  let height := side
  2 * Real.pi * radius * height = 400 * Real.pi :=
by
  let side := 20
  let radius := side / 2
  let height := side
  sorry

end NUMINAMATH_GPT_cylinder_lateral_surface_area_l1396_139684


namespace NUMINAMATH_GPT_sum_of_fractions_l1396_139687

def S_1 : List ℚ := List.range' 1 10 |>.map (λ n => n / 10)
def S_2 : List ℚ := List.replicate 4 (20 / 10)

def total_sum : ℚ := S_1.sum + S_2.sum

theorem sum_of_fractions : total_sum = 12.5 := by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l1396_139687


namespace NUMINAMATH_GPT_juice_m_smoothie_l1396_139626

/-- 
24 oz of juice p and 25 oz of juice v are mixed to make smoothies m and y. 
The ratio of p to v in smoothie m is 4 to 1 and that in y is 1 to 5. 
Prove that the amount of juice p in the smoothie m is 20 oz.
-/
theorem juice_m_smoothie (P_m P_y V_m V_y : ℕ)
  (h1 : P_m + P_y = 24)
  (h2 : V_m + V_y = 25)
  (h3 : 4 * V_m = P_m)
  (h4 : V_y = 5 * P_y) :
  P_m = 20 :=
sorry

end NUMINAMATH_GPT_juice_m_smoothie_l1396_139626


namespace NUMINAMATH_GPT_quadratic_root_k_eq_one_l1396_139623

theorem quadratic_root_k_eq_one
  (k : ℝ)
  (h₀ : (k + 3) ≠ 0)
  (h₁ : ∃ x : ℝ, (x = 0) ∧ ((k + 3) * x^2 + 5 * x + k^2 + 2 * k - 3 = 0)) :
  k = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_k_eq_one_l1396_139623


namespace NUMINAMATH_GPT_probability_largest_ball_is_six_l1396_139621

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_largest_ball_is_six : 
  (choose 6 4 : ℝ) / (choose 10 4 : ℝ) = (15 : ℝ) / (210 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_probability_largest_ball_is_six_l1396_139621


namespace NUMINAMATH_GPT_two_positive_numbers_inequality_three_positive_numbers_inequality_l1396_139695

theorem two_positive_numbers_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b) * (1 / a + 1 / b) ≥ 4 :=
by sorry

theorem three_positive_numbers_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
by sorry

end NUMINAMATH_GPT_two_positive_numbers_inequality_three_positive_numbers_inequality_l1396_139695


namespace NUMINAMATH_GPT_books_sold_l1396_139604

theorem books_sold (initial_books remaining_books sold_books : ℕ):
  initial_books = 33 → 
  remaining_books = 7 → 
  sold_books = initial_books - remaining_books → 
  sold_books = 26 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_books_sold_l1396_139604


namespace NUMINAMATH_GPT_correct_value_l1396_139648

theorem correct_value (x : ℕ) (h : 14 * x = 42) : 12 * x = 36 := by
  sorry

end NUMINAMATH_GPT_correct_value_l1396_139648


namespace NUMINAMATH_GPT_positive_expression_l1396_139616

theorem positive_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^2 * (b + c) + a * (b^2 + c^2 - b * c) > 0 :=
by sorry

end NUMINAMATH_GPT_positive_expression_l1396_139616


namespace NUMINAMATH_GPT_total_selling_price_l1396_139680

def selling_price_A (purchase_price_A : ℝ) : ℝ :=
  purchase_price_A - (0.15 * purchase_price_A)

def selling_price_B (purchase_price_B : ℝ) : ℝ :=
  purchase_price_B + (0.10 * purchase_price_B)

def selling_price_C (purchase_price_C : ℝ) : ℝ :=
  purchase_price_C - (0.05 * purchase_price_C)

theorem total_selling_price 
  (purchase_price_A : ℝ)
  (purchase_price_B : ℝ)
  (purchase_price_C : ℝ)
  (loss_A : ℝ := 0.15)
  (gain_B : ℝ := 0.10)
  (loss_C : ℝ := 0.05)
  (total_price := selling_price_A purchase_price_A + selling_price_B purchase_price_B + selling_price_C purchase_price_C) :
  purchase_price_A = 1400 → purchase_price_B = 2500 → purchase_price_C = 3200 →
  total_price = 6980 :=
by sorry

end NUMINAMATH_GPT_total_selling_price_l1396_139680


namespace NUMINAMATH_GPT_two_digit_number_is_91_l1396_139631

/-- A positive two-digit number is odd and is a multiple of 13.
    The product of its digits is a perfect square.
    What is this two-digit number? -/
theorem two_digit_number_is_91 (M : ℕ) (h1 : M > 9) (h2 : M < 100) (h3 : M % 2 = 1) (h4 : M % 13 = 0) (h5 : ∃ n : ℕ, n * n = (M / 10) * (M % 10)) :
  M = 91 :=
sorry

end NUMINAMATH_GPT_two_digit_number_is_91_l1396_139631


namespace NUMINAMATH_GPT_prove_b_minus_a_l1396_139611

noncomputable def point := (ℝ × ℝ)

def rotate90 (p : point) (c : point) : point :=
  let (x, y) := p
  let (h, k) := c
  (h - (y - k), k + (x - h))

def reflect_y_eq_x (p : point) : point :=
  let (x, y) := p
  (y, x)

def transformed_point (a b : ℝ) : point :=
  reflect_y_eq_x (rotate90 (a, b) (2, 6))

theorem prove_b_minus_a (a b : ℝ) (h1 : transformed_point a b = (-7, 4)) : b - a = 15 :=
by
  sorry

end NUMINAMATH_GPT_prove_b_minus_a_l1396_139611


namespace NUMINAMATH_GPT_sequence_terms_l1396_139683

theorem sequence_terms (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (hS : ∀ n, S n = 3 ^ n - 2) :
  (a 1 = 1) ∧ (∀ n, n ≥ 2 → a n = 2 * 3 ^ (n - 1)) := by
  sorry

end NUMINAMATH_GPT_sequence_terms_l1396_139683


namespace NUMINAMATH_GPT_triangular_array_sum_digits_l1396_139650

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 2145) : (N / 10 + N % 10) = 11 := 
sorry

end NUMINAMATH_GPT_triangular_array_sum_digits_l1396_139650


namespace NUMINAMATH_GPT_minimize_sum_of_squares_l1396_139612

open Real

-- Assume x, y are positive real numbers and x + y = s
variables {x y s : ℝ}
variables (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : x + y = s)

theorem minimize_sum_of_squares :
  (x = y) ∧ (2 * x * x = s * s / 2) → (x = s / 2 ∧ y = s / 2 ∧ x^2 + y^2 = s^2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_minimize_sum_of_squares_l1396_139612


namespace NUMINAMATH_GPT_exists_positive_m_dividing_f_100_l1396_139670

noncomputable def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_positive_m_dividing_f_100:
  ∃ (m : ℤ), m > 0 ∧ 19881 ∣ (3^100 * (m + 1) - 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_positive_m_dividing_f_100_l1396_139670


namespace NUMINAMATH_GPT_max_value_of_f_min_value_of_a2_4b2_min_value_of_a2_4b2_equals_l1396_139614

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + 2 * b|

theorem max_value_of_f (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∀ x, f x a b ≤ a + 2 * b :=
by sorry

theorem min_value_of_a2_4b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max : a + 2 * b = 1) :
  a^2 + 4 * b^2 ≥ 1 / 2 :=
by sorry

theorem min_value_of_a2_4b2_equals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max : a + 2 * b = 1) :
  ∃ a b, a = 1 / 2 ∧ b = 1 / 4 ∧ (a^2 + 4 * b^2 = 1 / 2) :=
by sorry

end NUMINAMATH_GPT_max_value_of_f_min_value_of_a2_4b2_min_value_of_a2_4b2_equals_l1396_139614


namespace NUMINAMATH_GPT_john_hourly_rate_with_bonus_l1396_139608

theorem john_hourly_rate_with_bonus:
  ∀ (daily_wage : ℝ) (work_hours : ℕ) (bonus : ℝ) (extra_hours : ℕ),
    daily_wage = 80 →
    work_hours = 8 →
    bonus = 20 →
    extra_hours = 2 →
    (daily_wage + bonus) / (work_hours + extra_hours) = 10 :=
by
  intros daily_wage work_hours bonus extra_hours
  intros h1 h2 h3 h4
  -- sorry: the proof is omitted
  sorry

end NUMINAMATH_GPT_john_hourly_rate_with_bonus_l1396_139608


namespace NUMINAMATH_GPT_builder_needs_boards_l1396_139674

theorem builder_needs_boards (packages : ℕ) (boards_per_package : ℕ) (total_boards : ℕ)
  (h1 : packages = 52)
  (h2 : boards_per_package = 3)
  (h3 : total_boards = packages * boards_per_package) : 
  total_boards = 156 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_builder_needs_boards_l1396_139674


namespace NUMINAMATH_GPT_alexis_dresses_l1396_139627

-- Definitions based on the conditions
def isabella_total : ℕ := 13
def alexis_total : ℕ := 3 * isabella_total
def alexis_pants : ℕ := 21

-- Theorem statement
theorem alexis_dresses : alexis_total - alexis_pants = 18 := by
  sorry

end NUMINAMATH_GPT_alexis_dresses_l1396_139627


namespace NUMINAMATH_GPT_simplify_expression_l1396_139647

variable (x y : ℝ)

theorem simplify_expression : 3 * y + 5 * y + 6 * y + 2 * x + 4 * x = 14 * y + 6 * x :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1396_139647


namespace NUMINAMATH_GPT_monthly_salary_l1396_139652

theorem monthly_salary (S : ℝ) (h1 : 0.20 * S + 1.20 * 0.80 * S = S) (h2 : S - 1.20 * 0.80 * S = 260) : S = 6500 :=
by
  sorry

end NUMINAMATH_GPT_monthly_salary_l1396_139652


namespace NUMINAMATH_GPT_students_not_opt_for_math_l1396_139635

theorem students_not_opt_for_math (total_students S E both_subjects M : ℕ) 
    (h1 : total_students = 40) 
    (h2 : S = 15) 
    (h3 : E = 2) 
    (h4 : both_subjects = 7) 
    (h5 : total_students - both_subjects = M + S - E) : M = 20 := 
  by
  sorry

end NUMINAMATH_GPT_students_not_opt_for_math_l1396_139635


namespace NUMINAMATH_GPT_number_of_sides_of_polygon_l1396_139671

theorem number_of_sides_of_polygon (exterior_angle : ℝ) (sum_exterior_angles : ℝ) (h1 : exterior_angle = 30) (h2 : sum_exterior_angles = 360) :
  sum_exterior_angles / exterior_angle = 12 := 
by
  sorry

end NUMINAMATH_GPT_number_of_sides_of_polygon_l1396_139671
