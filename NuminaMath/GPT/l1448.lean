import Mathlib

namespace NUMINAMATH_GPT_at_least_one_distinct_root_l1448_144807

theorem at_least_one_distinct_root {a b : ℝ} (ha : a > 4) (hb : b > 4) :
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + a * x₁ + b = 0 ∧ x₂^2 + a * x₂ + b = 0) ∨
    (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ y₁^2 + b * y₁ + a = 0 ∧ y₂^2 + b * y₂ + a = 0) :=
sorry

end NUMINAMATH_GPT_at_least_one_distinct_root_l1448_144807


namespace NUMINAMATH_GPT_monotonically_increasing_interval_l1448_144851

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem monotonically_increasing_interval :
  ∀ x, 0 < x ∧ x ≤ π / 6 → ∀ y, x ≤ y ∧ y < π / 2 → f x ≤ f y :=
by
  intro x hx y hy
  sorry

end NUMINAMATH_GPT_monotonically_increasing_interval_l1448_144851


namespace NUMINAMATH_GPT_gcd_3570_4840_l1448_144835

-- Define the numbers
def num1 : Nat := 3570
def num2 : Nat := 4840

-- Define the problem statement
theorem gcd_3570_4840 : Nat.gcd num1 num2 = 10 := by
  sorry

end NUMINAMATH_GPT_gcd_3570_4840_l1448_144835


namespace NUMINAMATH_GPT_smallest_log_log_x0_l1448_144844

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem smallest_log_log_x0 (x₀ : ℝ) (h₀ : f x₀ = 0) (h_dom : 2 < x₀ ∧ x₀ < Real.exp 1) :
  min (min (Real.log x₀) (Real.log (Real.sqrt x₀))) (min (Real.log (Real.log x₀)) ((Real.log x₀)^2)) = Real.log (Real.log x₀) :=
sorry

end NUMINAMATH_GPT_smallest_log_log_x0_l1448_144844


namespace NUMINAMATH_GPT_chime_date_is_march_22_2003_l1448_144829

-- Definitions
def clock_chime (n : ℕ) : ℕ := n % 12

def half_hour_chimes (half_hours : ℕ) : ℕ := half_hours
def hourly_chimes (hours : List ℕ) : ℕ := hours.map clock_chime |>.sum

-- Problem conditions and result
def initial_chimes_and_half_hours : ℕ := half_hour_chimes 9
def initial_hourly_chimes : ℕ := hourly_chimes [4, 5, 6, 7, 8, 9, 10, 11, 0]
def chimes_on_february_28_2003 : ℕ := initial_chimes_and_half_hours + initial_hourly_chimes

def half_hour_chimes_per_day : ℕ := half_hour_chimes 24
def hourly_chimes_per_day : ℕ := hourly_chimes (List.range 12 ++ List.range 12)
def total_chimes_per_day : ℕ := half_hour_chimes_per_day + hourly_chimes_per_day

def remaining_chimes_needed : ℕ := 2003 - chimes_on_february_28_2003
def full_days_needed : ℕ := remaining_chimes_needed / total_chimes_per_day
def additional_chimes_needed : ℕ := remaining_chimes_needed % total_chimes_per_day

-- Lean theorem statement
theorem chime_date_is_march_22_2003 :
    (full_days_needed = 21) → (additional_chimes_needed < total_chimes_per_day) → 
    true :=
by
  sorry

end NUMINAMATH_GPT_chime_date_is_march_22_2003_l1448_144829


namespace NUMINAMATH_GPT_Tim_income_percentage_less_than_Juan_l1448_144853

-- Definitions for the problem
variables (T M J : ℝ)

-- Conditions based on the problem
def condition1 : Prop := M = 1.60 * T
def condition2 : Prop := M = 0.80 * J

-- Goal statement
theorem Tim_income_percentage_less_than_Juan :
  condition1 T M ∧ condition2 M J → T = 0.50 * J :=
by sorry

end NUMINAMATH_GPT_Tim_income_percentage_less_than_Juan_l1448_144853


namespace NUMINAMATH_GPT_cole_drive_to_work_time_l1448_144895

variables (D : ℝ) (T_work T_home : ℝ)

def speed_work : ℝ := 75
def speed_home : ℝ := 105
def total_time : ℝ := 2

theorem cole_drive_to_work_time :
  (T_work = D / speed_work) ∧
  (T_home = D / speed_home) ∧
  (T_work + T_home = total_time) →
  T_work * 60 = 70 :=
by
  sorry

end NUMINAMATH_GPT_cole_drive_to_work_time_l1448_144895


namespace NUMINAMATH_GPT_every_integer_as_sum_of_squares_l1448_144861

theorem every_integer_as_sum_of_squares (n : ℤ) : ∃ x y z : ℕ, 0 < x ∧ 0 < y ∧ 0 < z ∧ n = (x^2 : ℤ) + (y^2 : ℤ) - (z^2 : ℤ) :=
by sorry

end NUMINAMATH_GPT_every_integer_as_sum_of_squares_l1448_144861


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1448_144862

theorem sufficient_but_not_necessary (x : ℝ) : ((0 < x) → (|x-1| - |x| ≤ 1)) ∧ ((|x-1| - |x| ≤ 1) → True) ∧ ¬((|x-1| - |x| ≤ 1) → (0 < x)) := sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1448_144862


namespace NUMINAMATH_GPT_min_value_expression_l1448_144810

theorem min_value_expression : 
  ∃ x : ℝ, ∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) ≥ (15 - x) * (8 - x) * (15 + x) * (8 + x) ∧ 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6480.25 :=
by sorry

end NUMINAMATH_GPT_min_value_expression_l1448_144810


namespace NUMINAMATH_GPT_repeating_decimal_sum_l1448_144818

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l1448_144818


namespace NUMINAMATH_GPT_greatest_divisor_remainders_l1448_144887

theorem greatest_divisor_remainders (d : ℤ) :
  d > 0 → (1657 % d = 10) → (2037 % d = 7) → d = 1 :=
by
  intros hdg h1657 h2037
  sorry

end NUMINAMATH_GPT_greatest_divisor_remainders_l1448_144887


namespace NUMINAMATH_GPT_cost_of_each_soda_l1448_144842

theorem cost_of_each_soda (total_cost sandwiches_cost : ℝ) (number_of_sodas : ℕ)
  (h_total_cost : total_cost = 6.46)
  (h_sandwiches_cost : sandwiches_cost = 2 * 1.49) :
  total_cost - sandwiches_cost = 4 * 0.87 := by
  sorry

end NUMINAMATH_GPT_cost_of_each_soda_l1448_144842


namespace NUMINAMATH_GPT_average_speed_train_l1448_144819

theorem average_speed_train (d1 d2 : ℝ) (t1 t2 : ℝ) 
  (h_d1 : d1 = 325) (h_d2 : d2 = 470)
  (h_t1 : t1 = 3.5) (h_t2 : t2 = 4) :
  (d1 + d2) / (t1 + t2) = 106 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_train_l1448_144819


namespace NUMINAMATH_GPT_stadium_capacity_l1448_144874

theorem stadium_capacity 
  (C : ℕ)
  (entry_fee : ℕ := 20)
  (three_fourth_full_fees : ℕ := 3 / 4 * C * entry_fee)
  (full_fees : ℕ := C * entry_fee)
  (fee_difference : ℕ := full_fees - three_fourth_full_fees)
  (h : fee_difference = 10000) :
  C = 2000 :=
by
  sorry

end NUMINAMATH_GPT_stadium_capacity_l1448_144874


namespace NUMINAMATH_GPT_angle_in_triangle_l1448_144800

theorem angle_in_triangle
  (A B C : Type)
  (a b c : ℝ)
  (angle_ABC : ℝ)
  (h1 : a = 15)
  (h2 : angle_ABC = π/3 ∨ angle_ABC = 2 * π / 3)
  : angle_ABC = π/3 ∨ angle_ABC = 2 * π / 3 := 
  sorry

end NUMINAMATH_GPT_angle_in_triangle_l1448_144800


namespace NUMINAMATH_GPT_intervals_of_increase_l1448_144834

def f (x : ℝ) : ℝ := 2*x^3 - 6*x^2 + 7

theorem intervals_of_increase : 
  ∀ x : ℝ, (x < 0 ∨ x > 2) → (6*x^2 - 12*x > 0) :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_intervals_of_increase_l1448_144834


namespace NUMINAMATH_GPT_volume_cone_equals_cylinder_minus_surface_area_l1448_144837

theorem volume_cone_equals_cylinder_minus_surface_area (r h : ℝ) :
  let V_cyl := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * h
  let S_lateral_cyl := 2 * π * r * h
  V_cone = V_cyl - (1 / 3) * S_lateral_cyl * r := by
  let V_cyl := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * h
  let S_lateral_cyl := 2 * π * r * h
  sorry

end NUMINAMATH_GPT_volume_cone_equals_cylinder_minus_surface_area_l1448_144837


namespace NUMINAMATH_GPT_total_income_l1448_144871

variable (I : ℝ)

/-- A person distributed 20% of his income to his 3 children each. -/
def distributed_children (I : ℝ) : ℝ := 3 * 0.20 * I

/-- He deposited 30% of his income to his wife's account. -/
def deposited_wife (I : ℝ) : ℝ := 0.30 * I

/-- The total percentage of his income that was given away is 90%. -/
def total_given_away (I : ℝ) : ℝ := distributed_children I + deposited_wife I 

/-- The remaining income after giving away 90%. -/
def remaining_income (I : ℝ) : ℝ := I - total_given_away I

/-- He donated 5% of the remaining income to the orphan house. -/
def donated_orphan_house (remaining : ℝ) : ℝ := 0.05 * remaining

/-- Finally, he has $40,000 left, which is 95% of the remaining income. -/
def final_amount (remaining : ℝ) : ℝ := 0.95 * remaining

theorem total_income (I : ℝ) (h : final_amount (remaining_income I) = 40000) :
  I = 421052.63 := 
  sorry

end NUMINAMATH_GPT_total_income_l1448_144871


namespace NUMINAMATH_GPT_largest_divisor_of_n4_minus_n2_l1448_144815

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 12 ∣ (n^4 - n^2) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_n4_minus_n2_l1448_144815


namespace NUMINAMATH_GPT_trajectory_eq_l1448_144833

theorem trajectory_eq :
  ∀ (x y : ℝ), abs x * abs y = 1 → (x * y = 1 ∨ x * y = -1) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_trajectory_eq_l1448_144833


namespace NUMINAMATH_GPT_probability_area_l1448_144890

noncomputable def probability_x_y_le_five (x y : ℝ) : ℚ :=
  if 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8 ∧ x + y ≤ 5 then 1 else 0

theorem probability_area {P : ℚ} :
  (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8 → P = probability_x_y_le_five x y / (4 * 8)) →
  P = 5 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_area_l1448_144890


namespace NUMINAMATH_GPT_total_profit_or_loss_is_negative_175_l1448_144867

theorem total_profit_or_loss_is_negative_175
    (price_A price_B selling_price : ℝ)
    (profit_A loss_B : ℝ)
    (h1 : selling_price = 2100)
    (h2 : profit_A = 0.2)
    (h3 : loss_B = 0.2)
    (hA : price_A * (1 + profit_A) = selling_price)
    (hB : price_B * (1 - loss_B) = selling_price) :
    (selling_price + selling_price) - (price_A + price_B) = -175 := 
by 
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_total_profit_or_loss_is_negative_175_l1448_144867


namespace NUMINAMATH_GPT_least_value_of_x_l1448_144817

theorem least_value_of_x 
  (x : ℕ) (p : ℕ) 
  (h1 : x > 0) 
  (h2 : Prime p) 
  (h3 : ∃ q, Prime q ∧ q % 2 = 1 ∧ x = 9 * p * q) : 
  x = 90 := 
sorry

end NUMINAMATH_GPT_least_value_of_x_l1448_144817


namespace NUMINAMATH_GPT_ken_gave_manny_10_pencils_l1448_144859

theorem ken_gave_manny_10_pencils (M : ℕ) 
  (ken_pencils : ℕ := 50)
  (ken_kept : ℕ := 20)
  (ken_distributed : ℕ := ken_pencils - ken_kept)
  (nilo_pencils : ℕ := M + 10)
  (distribution_eq : M + nilo_pencils = ken_distributed) : 
  M = 10 :=
by
  sorry

end NUMINAMATH_GPT_ken_gave_manny_10_pencils_l1448_144859


namespace NUMINAMATH_GPT_arithmetic_expression_l1448_144827

theorem arithmetic_expression : 5 + 12 / 3 - 3 ^ 2 + 1 = 1 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_l1448_144827


namespace NUMINAMATH_GPT_find_locus_of_P_l1448_144848

theorem find_locus_of_P:
  ∃ x y: ℝ, (x - 1)^2 + y^2 = 9 ∧ y ≠ 0 ∧
          ((x + 2)^2 + y^2 + (x - 4)^2 + y^2 = 36) :=
sorry

end NUMINAMATH_GPT_find_locus_of_P_l1448_144848


namespace NUMINAMATH_GPT_sum_of_x_intersections_is_zero_l1448_144836

-- Definition of an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Definition for the x-coordinates of the intersection points with x-axis
def intersects_x_axis (f : ℝ → ℝ) (x_coords : List ℝ) : Prop :=
  (∀ x ∈ x_coords, f x = 0) ∧ (x_coords.length = 4)

-- Main theorem
theorem sum_of_x_intersections_is_zero 
  (f : ℝ → ℝ)
  (x_coords : List ℝ)
  (h1 : is_even_function f)
  (h2 : intersects_x_axis f x_coords) : 
  x_coords.sum = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_x_intersections_is_zero_l1448_144836


namespace NUMINAMATH_GPT_percentage_mutant_frogs_is_33_l1448_144826

def num_extra_legs_frogs := 5
def num_two_heads_frogs := 2
def num_bright_red_frogs := 2
def num_normal_frogs := 18

def total_mutant_frogs := num_extra_legs_frogs + num_two_heads_frogs + num_bright_red_frogs
def total_frogs := total_mutant_frogs + num_normal_frogs

theorem percentage_mutant_frogs_is_33 :
  Float.round (100 * total_mutant_frogs.toFloat / total_frogs.toFloat) = 33 :=
by 
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_percentage_mutant_frogs_is_33_l1448_144826


namespace NUMINAMATH_GPT_max_vehicles_div_10_l1448_144821

-- Each vehicle is 5 meters long
def vehicle_length : ℕ := 5

-- The speed rule condition
def speed_rule (m : ℕ) : ℕ := 20 * m

-- Maximum number of vehicles in one hour
def max_vehicles_per_hour (m : ℕ) : ℕ := 4000 * m / (m + 1)

-- N is the maximum whole number of vehicles
def N : ℕ := 4000

-- The target statement to prove: quotient when N is divided by 10
theorem max_vehicles_div_10 : N / 10 = 400 :=
by
  -- Definitions and given conditions go here
  sorry

end NUMINAMATH_GPT_max_vehicles_div_10_l1448_144821


namespace NUMINAMATH_GPT_yan_distance_ratio_l1448_144873

theorem yan_distance_ratio (w x y : ℝ) (h1 : w > 0) (h2 : x > 0) (h3 : y > 0)
(h4 : y / w = x / w + (x + y) / (5 * w)) : x / y = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_yan_distance_ratio_l1448_144873


namespace NUMINAMATH_GPT_darius_scores_less_l1448_144847

variable (D M Ma : ℕ)

-- Conditions
def condition1 := D = 10
def condition2 := Ma = D + 3
def condition3 := D + M + Ma = 38

-- Theorem to prove
theorem darius_scores_less (D M Ma : ℕ) (h1 : condition1 D) (h2 : condition2 D Ma) (h3 : condition3 D M Ma) : M - D = 5 :=
by
  sorry

end NUMINAMATH_GPT_darius_scores_less_l1448_144847


namespace NUMINAMATH_GPT_minimal_height_exists_l1448_144897

noncomputable def height_min_material (x : ℝ) : ℝ := 4 / (x^2)

theorem minimal_height_exists
  (x h : ℝ)
  (volume_cond : x^2 * h = 4)
  (surface_area_cond : h = height_min_material x) :
  h = 1 := by
  sorry

end NUMINAMATH_GPT_minimal_height_exists_l1448_144897


namespace NUMINAMATH_GPT_largest_cube_volume_l1448_144878

theorem largest_cube_volume (width length height : ℕ) (h₁ : width = 15) (h₂ : length = 12) (h₃ : height = 8) :
  ∃ V, V = 512 := by
  use 8^3
  sorry

end NUMINAMATH_GPT_largest_cube_volume_l1448_144878


namespace NUMINAMATH_GPT_purely_imaginary_m_complex_division_a_plus_b_l1448_144845

-- Problem 1: Prove that m=-2 for z to be purely imaginary
theorem purely_imaginary_m (m : ℝ) (h : ∀ z : ℂ, z = (m - 1) * (m + 2) + (m - 1) * I → z.im = z.im) : m = -2 :=
sorry

-- Problem 2: Prove a+b = 13/10 with given conditions
theorem complex_division_a_plus_b (a b : ℝ) (m : ℝ) (h_m : m = 2) 
  (h_z : z = 4 + I) (h_eq : (z + I) / (z - I) = a + b * I) : a + b = 13 / 10 :=
sorry

end NUMINAMATH_GPT_purely_imaginary_m_complex_division_a_plus_b_l1448_144845


namespace NUMINAMATH_GPT_a2b_sub_ab2_eq_neg16sqrt5_l1448_144868

noncomputable def a : ℝ := 4 + 2 * Real.sqrt 5
noncomputable def b : ℝ := 4 - 2 * Real.sqrt 5

theorem a2b_sub_ab2_eq_neg16sqrt5 : a^2 * b - a * b^2 = -16 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_a2b_sub_ab2_eq_neg16sqrt5_l1448_144868


namespace NUMINAMATH_GPT_solve_for_y_l1448_144811

theorem solve_for_y (y : ℝ) (h_sum : (1 + 99) * 99 / 2 = 4950)
  (h_avg : (4950 + y) / 100 = 50 * y) : y = 4950 / 4999 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1448_144811


namespace NUMINAMATH_GPT_stormi_mowing_charge_l1448_144866

theorem stormi_mowing_charge (cars_washed : ℕ) (car_wash_price : ℕ) (lawns_mowed : ℕ) (bike_cost : ℕ) (money_needed_more : ℕ) 
  (total_from_cars : ℕ := cars_washed * car_wash_price)
  (total_earned : ℕ := bike_cost - money_needed_more)
  (earned_from_lawns : ℕ := total_earned - total_from_cars) :
  cars_washed = 3 → car_wash_price = 10 → lawns_mowed = 2 → bike_cost = 80 → money_needed_more = 24 → earned_from_lawns / lawns_mowed = 13 := 
by
  sorry

end NUMINAMATH_GPT_stormi_mowing_charge_l1448_144866


namespace NUMINAMATH_GPT_count_perfect_cubes_between_bounds_l1448_144876

theorem count_perfect_cubes_between_bounds :
  let lower_bound := 3^6 + 1
  let upper_bound := 3^12 + 1
  -- the number of perfect cubes k^3 such that 3^6 + 1 < k^3 < 3^12 + 1 inclusive is 72
  (730 < k * k * k ∧ k * k * k <= 531442 ∧ 10 <= k ∧ k <= 81 → k = 72) :=
by
  let lower_bound : ℕ := 3^6 + 1
  let upper_bound : ℕ := 3^12 + 1
  sorry

end NUMINAMATH_GPT_count_perfect_cubes_between_bounds_l1448_144876


namespace NUMINAMATH_GPT_solve_cubic_root_eq_l1448_144820

theorem solve_cubic_root_eq (x : ℝ) : (∃ x, 3 - x / 3 = -8) -> x = 33 :=
by
  sorry

end NUMINAMATH_GPT_solve_cubic_root_eq_l1448_144820


namespace NUMINAMATH_GPT_largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7_l1448_144864

-- Definitions and conditions
def binomial_expansion (x : ℕ) (n : ℕ) : ℕ := (x + (3 * x^2))^n

-- Problem statements
theorem largest_binomial_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  (2^n = 128) →
  ∃ t : ℕ, t = 2835 * x^11 := 
by sorry

theorem largest_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  exists t, t = 5103 * x^13 :=
by sorry

theorem remainder_mod_7 :
  ∀ x n,
  x = 3 →
  n = 2016 →
  (x + (3 * x^2))^n % 7 = 1 :=
by sorry

end NUMINAMATH_GPT_largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7_l1448_144864


namespace NUMINAMATH_GPT_smallest_multiple_9_11_13_l1448_144801

theorem smallest_multiple_9_11_13 : ∃ n : ℕ, n > 0 ∧ (9 ∣ n) ∧ (11 ∣ n) ∧ (13 ∣ n) ∧ n = 1287 := 
 by {
   sorry
 }

end NUMINAMATH_GPT_smallest_multiple_9_11_13_l1448_144801


namespace NUMINAMATH_GPT_tangent_line_circle_intersection_l1448_144832

open Real

noncomputable def is_tangent (θ : ℝ) : Prop :=
  abs (4 * tan θ) / sqrt ((tan θ) ^ 2 + 1) = 2

theorem tangent_line_circle_intersection (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < π) :
  is_tangent θ ↔ θ = π / 6 ∨ θ = 5 * π / 6 :=
sorry

end NUMINAMATH_GPT_tangent_line_circle_intersection_l1448_144832


namespace NUMINAMATH_GPT_minimize_tank_construction_cost_l1448_144802

noncomputable def minimum_cost (l w h : ℝ) (P_base P_wall : ℝ) : ℝ :=
  P_base * (l * w) + P_wall * (2 * h * (l + w))

theorem minimize_tank_construction_cost :
  ∃ l w : ℝ, l * w = 9 ∧ l = w ∧
  minimum_cost l w 2 200 150 = 5400 :=
by
  sorry

end NUMINAMATH_GPT_minimize_tank_construction_cost_l1448_144802


namespace NUMINAMATH_GPT_gcd_and_lcm_of_18_and_24_l1448_144808

-- Definitions of gcd and lcm for the problem's context
def my_gcd (a b : ℕ) : ℕ := a.gcd b
def my_lcm (a b : ℕ) : ℕ := a.lcm b

-- Constants given in the problem
def a := 18
def b := 24

-- Proof problem statement
theorem gcd_and_lcm_of_18_and_24 : my_gcd a b = 6 ∧ my_lcm a b = 72 := by
  sorry

end NUMINAMATH_GPT_gcd_and_lcm_of_18_and_24_l1448_144808


namespace NUMINAMATH_GPT_circle_center_coordinates_l1448_144812

theorem circle_center_coordinates :
  ∃ c : ℝ × ℝ, (∀ x y : ℝ, x^2 + y^2 - x + 2*y = 0 ↔ (x-c.1)^2 + (y-c.2)^2 = (5/4)) ∧ c = (1/2, -1) :=
sorry

end NUMINAMATH_GPT_circle_center_coordinates_l1448_144812


namespace NUMINAMATH_GPT_male_students_count_l1448_144875

theorem male_students_count
  (average_all_students : ℕ → ℕ → ℚ → Prop)
  (average_male_students : ℕ → ℚ → Prop)
  (average_female_students : ℕ → ℚ → Prop)
  (F : ℕ)
  (total_average : average_all_students (F + M) (83 * M + 92 * F) 90)
  (male_average : average_male_students M 83)
  (female_average : average_female_students 28 92) :
  ∃ (M : ℕ), M = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_male_students_count_l1448_144875


namespace NUMINAMATH_GPT_Roberto_outfit_count_l1448_144850

theorem Roberto_outfit_count :
  let trousers := 5
  let shirts := 6
  let jackets := 3
  let ties := 2
  trousers * shirts * jackets * ties = 180 :=
by
  sorry

end NUMINAMATH_GPT_Roberto_outfit_count_l1448_144850


namespace NUMINAMATH_GPT_train_length_l1448_144841

theorem train_length (speed_first_train speed_second_train : ℝ) (length_second_train : ℝ) (cross_time : ℝ) (L1 : ℝ) : 
  speed_first_train = 100 ∧ 
  speed_second_train = 60 ∧ 
  length_second_train = 300 ∧ 
  cross_time = 18 → 
  L1 = 420 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1448_144841


namespace NUMINAMATH_GPT_power_of_7_mod_10_l1448_144846

theorem power_of_7_mod_10 (k : ℕ) (h : 7^4 ≡ 1 [MOD 10]) : 7^150 ≡ 9 [MOD 10] :=
sorry

end NUMINAMATH_GPT_power_of_7_mod_10_l1448_144846


namespace NUMINAMATH_GPT_range_of_a_l1448_144843

theorem range_of_a (x : ℝ) (h : 1 < x) : ∀ a, (∀ x, 1 < x → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
sorry

end NUMINAMATH_GPT_range_of_a_l1448_144843


namespace NUMINAMATH_GPT_range_of_f_l1448_144888

def f (x : ℤ) := x + 1

theorem range_of_f : 
  (∀ x ∈ ({-1, 0, 1, 2} : Set ℤ), f x ∈ ({0, 1, 2, 3} : Set ℤ)) ∧ 
  (∀ y ∈ ({0, 1, 2, 3} : Set ℤ), ∃ x ∈ ({-1, 0, 1, 2} : Set ℤ), f x = y) := 
by 
  sorry

end NUMINAMATH_GPT_range_of_f_l1448_144888


namespace NUMINAMATH_GPT_chandler_weeks_to_save_l1448_144856

theorem chandler_weeks_to_save :
  let birthday_money := 50 + 35 + 15 + 20
  let weekly_earnings := 18
  let bike_cost := 650
  ∃ x : ℕ, (birthday_money + x * weekly_earnings) ≥ bike_cost ∧ (birthday_money + (x - 1) * weekly_earnings) < bike_cost := 
by
  sorry

end NUMINAMATH_GPT_chandler_weeks_to_save_l1448_144856


namespace NUMINAMATH_GPT_value_of_b_l1448_144891

theorem value_of_b (a b : ℕ) (q : ℝ)
  (h1 : q = 0.5)
  (h2 : a = 2020)
  (h3 : q = a / b) : b = 4040 := by
  sorry

end NUMINAMATH_GPT_value_of_b_l1448_144891


namespace NUMINAMATH_GPT_prop_D_l1448_144854

variable (a b : ℝ)

theorem prop_D (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
  by
    sorry

end NUMINAMATH_GPT_prop_D_l1448_144854


namespace NUMINAMATH_GPT_B_A_equals_expectedBA_l1448_144894

noncomputable def MatrixA : Matrix (Fin 2) (Fin 2) ℝ := sorry
noncomputable def MatrixB : Matrix (Fin 2) (Fin 2) ℝ := sorry
def MatrixAB : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 1], ![-2, 4]]
def expectedBA : Matrix (Fin 2) (Fin 2) ℝ := ![![10, 2], ![-4, 8]]

theorem B_A_equals_expectedBA (A B : Matrix (Fin 2) (Fin 2) ℝ)
  (h1 : A + B = 2 * A * B)
  (h2 : A * B = MatrixAB) : 
  B * A = expectedBA := by
  sorry

end NUMINAMATH_GPT_B_A_equals_expectedBA_l1448_144894


namespace NUMINAMATH_GPT_cafe_purchase_l1448_144839

theorem cafe_purchase (s d : ℕ) (h_d : d ≥ 2) (h_cost : 5 * s + 125 * d = 4000) :  s + d = 11 :=
    -- Proof steps go here
    sorry

end NUMINAMATH_GPT_cafe_purchase_l1448_144839


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_l1448_144885

variable (a x : ℝ)

theorem inequality_holds_for_all_x (h : a ∈ Set.Ioc (-2 : ℝ) 4): ∀ x : ℝ, (x^2 - a*x + 9 > 0) :=
sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_l1448_144885


namespace NUMINAMATH_GPT_find_f_values_l1448_144886

def func_property1 (f : ℕ → ℕ) : Prop := 
  ∀ a b : ℕ, a ≠ b → a * f a + b * f b > a * f b + b * f a

def func_property2 (f : ℕ → ℕ) : Prop := 
  ∀ n : ℕ, f (f n) = 3 * n

theorem find_f_values (f : ℕ → ℕ) (h1 : func_property1 f) (h2 : func_property2 f) : 
  f 1 + f 6 + f 28 = 66 :=
sorry

end NUMINAMATH_GPT_find_f_values_l1448_144886


namespace NUMINAMATH_GPT_first_book_length_l1448_144881

-- Statement of the problem
theorem first_book_length
  (x : ℕ) -- Number of pages in the first book
  (total_pages : ℕ)
  (days_in_two_weeks : ℕ)
  (pages_per_day : ℕ)
  (second_book_pages : ℕ := 100) :
  pages_per_day = 20 ∧ days_in_two_weeks = 14 ∧ total_pages = 280 ∧ total_pages = pages_per_day * days_in_two_weeks ∧ total_pages = x + second_book_pages → x = 180 :=
by
  sorry

end NUMINAMATH_GPT_first_book_length_l1448_144881


namespace NUMINAMATH_GPT_four_consecutive_integers_product_2520_l1448_144805

theorem four_consecutive_integers_product_2520 {a b c d : ℕ}
  (h1 : a + 1 = b) 
  (h2 : b + 1 = c) 
  (h3 : c + 1 = d) 
  (h4 : a * b * c * d = 2520) : 
  a = 6 := 
sorry

end NUMINAMATH_GPT_four_consecutive_integers_product_2520_l1448_144805


namespace NUMINAMATH_GPT_no_rational_roots_of_odd_l1448_144838

theorem no_rational_roots_of_odd (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) : ¬ ∃ x : ℚ, x^2 + 2 * m * x + 2 * n = 0 :=
sorry

end NUMINAMATH_GPT_no_rational_roots_of_odd_l1448_144838


namespace NUMINAMATH_GPT_range_of_u_l1448_144831

variable (a b u : ℝ)

theorem range_of_u (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (9 / b) = 1) : u ≤ 16 :=
by
  sorry

end NUMINAMATH_GPT_range_of_u_l1448_144831


namespace NUMINAMATH_GPT_intersection_eq_l1448_144899

def A : Set ℕ := {1, 2, 4, 6, 8}
def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem intersection_eq : A ∩ B = {2, 4, 8} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1448_144899


namespace NUMINAMATH_GPT_expression_range_l1448_144870

open Real -- Open the real number namespace

theorem expression_range (x y : ℝ) (h : (x - 1)^2 + (y - 4)^2 = 1) : 
  0 ≤ (x * y - x) / (x^2 + (y - 1)^2) ∧ (x * y - x) / (x^2 + (y - 1)^2) ≤ 12 / 25 :=
sorry -- Proof to be filled in.

end NUMINAMATH_GPT_expression_range_l1448_144870


namespace NUMINAMATH_GPT_candy_mixture_problem_l1448_144816

theorem candy_mixture_problem:
  ∃ x y : ℝ, x + y = 5 ∧ 3.20 * x + 1.70 * y = 10 ∧ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_candy_mixture_problem_l1448_144816


namespace NUMINAMATH_GPT_knicks_eq_knocks_l1448_144892

theorem knicks_eq_knocks :
  (∀ (k n : ℕ), 5 * k = 3 * n ∧ 4 * n = 6 * 36) →
  (∃ m : ℕ, 36 * m = 40 * k) :=
by
  sorry

end NUMINAMATH_GPT_knicks_eq_knocks_l1448_144892


namespace NUMINAMATH_GPT_twice_minus_three_algebraic_l1448_144825

def twice_minus_three (x : ℝ) : ℝ := 2 * x - 3

theorem twice_minus_three_algebraic (x : ℝ) : 
  twice_minus_three x = 2 * x - 3 :=
by sorry

end NUMINAMATH_GPT_twice_minus_three_algebraic_l1448_144825


namespace NUMINAMATH_GPT_union_A_B_range_of_a_l1448_144828

-- Definitions of sets A, B, and C
def A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 9 }
def B : Set ℝ := { x | 2 < x ∧ x < 5 }
def C (a : ℝ) : Set ℝ := { x | x > a }

-- Problem 1: Proving A ∪ B = { x | 2 < x ≤ 9 }
theorem union_A_B : A ∪ B = { x | 2 < x ∧ x ≤ 9 } :=
sorry

-- Problem 2: Proving the range of 'a' given B ∩ C = ∅
theorem range_of_a (a : ℝ) (h : B ∩ C a = ∅) : a ≥ 5 :=
sorry

end NUMINAMATH_GPT_union_A_B_range_of_a_l1448_144828


namespace NUMINAMATH_GPT_joey_read_percentage_l1448_144860

theorem joey_read_percentage : 
  ∀ (total_pages read_after_break : ℕ), 
  total_pages = 30 → read_after_break = 9 → 
  ( (total_pages - read_after_break : ℕ) / (total_pages : ℕ) * 100 ) = 70 :=
by
  intros total_pages read_after_break h_total h_after
  sorry

end NUMINAMATH_GPT_joey_read_percentage_l1448_144860


namespace NUMINAMATH_GPT_meadow_grazing_days_l1448_144869

theorem meadow_grazing_days 
    (a b x : ℝ) 
    (h1 : a + 6 * b = 27 * 6 * x)
    (h2 : a + 9 * b = 23 * 9 * x)
    : ∃ y : ℝ, (a + y * b = 21 * y * x) ∧ y = 12 := 
by
    sorry

end NUMINAMATH_GPT_meadow_grazing_days_l1448_144869


namespace NUMINAMATH_GPT_total_highlighters_l1448_144855

def num_pink_highlighters := 9
def num_yellow_highlighters := 8
def num_blue_highlighters := 5

theorem total_highlighters : 
  num_pink_highlighters + num_yellow_highlighters + num_blue_highlighters = 22 :=
by
  sorry

end NUMINAMATH_GPT_total_highlighters_l1448_144855


namespace NUMINAMATH_GPT_neg_or_implication_l1448_144803

theorem neg_or_implication {p q : Prop} : ¬(p ∨ q) → (¬p ∧ ¬q) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_neg_or_implication_l1448_144803


namespace NUMINAMATH_GPT_circle_circumference_ratio_l1448_144849

theorem circle_circumference_ratio (A₁ A₂ : ℝ) (h : A₁ / A₂ = 16 / 25) :
  ∃ C₁ C₂ : ℝ, (C₁ / C₂ = 4 / 5) :=
by
  -- Definitions and calculations to be done here
  sorry

end NUMINAMATH_GPT_circle_circumference_ratio_l1448_144849


namespace NUMINAMATH_GPT_no_positive_integers_satisfy_l1448_144880

theorem no_positive_integers_satisfy (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ¬ (3 * a^2 = b^2 + 1) := 
sorry

end NUMINAMATH_GPT_no_positive_integers_satisfy_l1448_144880


namespace NUMINAMATH_GPT_fifteen_pow_mn_eq_PnQm_l1448_144840

-- Definitions
def P (m : ℕ) := 3^m
def Q (n : ℕ) := 5^n

-- Theorem statement
theorem fifteen_pow_mn_eq_PnQm (m n : ℕ) : 15^(m * n) = (P m)^n * (Q n)^m :=
by
  -- Placeholder for the proof, which isn't required
  sorry

end NUMINAMATH_GPT_fifteen_pow_mn_eq_PnQm_l1448_144840


namespace NUMINAMATH_GPT_part_1_part_2_l1448_144823

-- Define proposition p
def proposition_p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (1 : ℝ) ∧ (x^2 - (a + 2) * x + 2 * a = 0)

-- Proposition q: x₁ and x₂ are two real roots of the equation x^2 - 2mx - 3 = 0
def proposition_q (m x₁ x₂ : ℝ) : Prop :=
  x₁ ^ 2 - 2 * m * x₁ - 3 = 0 ∧ x₂ ^ 2 - 2 * m * x₂ - 3 = 0

-- Inequality condition
def inequality_condition (a m x₁ x₂ : ℝ) : Prop :=
  a ^ 2 - 3 * a ≥ abs (x₁ - x₂)

-- Part 1: If proposition p is true, find the range of the real number a
theorem part_1 (a : ℝ) (h_p : proposition_p a) : -1 < a ∧ a < 1 :=
  sorry

-- Part 2: If exactly one of propositions p or q is true, find the range of the real number a
theorem part_2 (a m x₁ x₂ : ℝ) (h_p_or_q : (proposition_p a ∧ ¬(proposition_q m x₁ x₂)) ∨ (¬(proposition_p a) ∧ (proposition_q m x₁ x₂))) : (a < 1) ∨ (a ≥ 4) :=
  sorry

end NUMINAMATH_GPT_part_1_part_2_l1448_144823


namespace NUMINAMATH_GPT_line_passes_fixed_point_l1448_144857

theorem line_passes_fixed_point (a b : ℝ) (h : a + 2 * b = 1) : 
  a * (1/2) + 3 * (-1/6) + b = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_passes_fixed_point_l1448_144857


namespace NUMINAMATH_GPT_percentage_of_first_to_second_l1448_144882

theorem percentage_of_first_to_second (X : ℝ) (first second : ℝ) :
  first = 0.06 * X →
  second = 0.30 * X →
  (first / second) * 100 = 20 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_percentage_of_first_to_second_l1448_144882


namespace NUMINAMATH_GPT_prove_range_of_a_l1448_144863

noncomputable def f (x a : ℝ) := x^2 + (a + 1) * x + Real.log (abs (a + 2))

def is_increasing (f : ℝ → ℝ) (interval : Set ℝ) :=
 ∀ ⦃x y⦄, x ∈ interval → y ∈ interval → x ≤ y → f x ≤ f y

def g (x a : ℝ) := (a + 1) * x
def is_decreasing (g : ℝ → ℝ) :=
 ∀ ⦃x y⦄, x ≤ y → g y ≤ g x

def proposition_p (a : ℝ) : Prop :=
  is_increasing (f a) (Set.Ici ((a + 1)^2))

def proposition_q (a : ℝ) : Prop :=
  is_decreasing (g a)

theorem prove_range_of_a (a : ℝ) (h : ¬ (proposition_p a ↔ proposition_q a)) :
  a > -3 / 2 :=
sorry

end NUMINAMATH_GPT_prove_range_of_a_l1448_144863


namespace NUMINAMATH_GPT_intersection_point_l1448_144893

theorem intersection_point : 
  ∃ (x y : ℚ), y = - (5/3 : ℚ) * x ∧ y + 3 = 15 * x - 6 ∧ x = 27 / 50 ∧ y = - 9 / 10 := 
by
  sorry

end NUMINAMATH_GPT_intersection_point_l1448_144893


namespace NUMINAMATH_GPT_remainder_845307_div_6_l1448_144858

theorem remainder_845307_div_6 :
  let n := 845307
  ∃ r : ℕ, n % 6 = r ∧ r = 3 :=
by
  let n := 845307
  have h_div_2 : ¬(n % 2 = 0) := by sorry
  have h_div_3 : n % 3 = 0 := by sorry
  exact ⟨3, by sorry, rfl⟩

end NUMINAMATH_GPT_remainder_845307_div_6_l1448_144858


namespace NUMINAMATH_GPT_theater_price_balcony_l1448_144872

theorem theater_price_balcony 
  (price_orchestra : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (extra_balcony_tickets : ℕ) (price_balcony : ℕ) 
  (h1 : price_orchestra = 12) 
  (h2 : total_tickets = 380) 
  (h3 : total_revenue = 3320) 
  (h4 : extra_balcony_tickets = 240) 
  (h5 : ∃ (O : ℕ), O + (O + extra_balcony_tickets) = total_tickets ∧ (price_orchestra * O) + (price_balcony * (O + extra_balcony_tickets)) = total_revenue) : 
  price_balcony = 8 := 
by
  sorry

end NUMINAMATH_GPT_theater_price_balcony_l1448_144872


namespace NUMINAMATH_GPT_smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7_l1448_144889

theorem smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7 :
  ∃ n : ℕ, n % 45 = 0 ∧ (n - 100) % 7 = 0 ∧ n = 135 :=
sorry

end NUMINAMATH_GPT_smallest_number_multiple_of_45_and_exceeds_100_by_multiple_of_7_l1448_144889


namespace NUMINAMATH_GPT_original_pencils_l1448_144896

-- Define the conditions given in the problem
variable (total_pencils_now : ℕ) [DecidableEq ℕ] (pencils_by_Mike : ℕ)

-- State the problem to prove
theorem original_pencils (h1 : total_pencils_now = 71) (h2 : pencils_by_Mike = 30) : total_pencils_now - pencils_by_Mike = 41 := by
  sorry

end NUMINAMATH_GPT_original_pencils_l1448_144896


namespace NUMINAMATH_GPT_english_homework_correct_time_l1448_144822

-- Define the given conditions as constants
def total_time : ℕ := 180 -- 3 hours in minutes
def math_homework_time : ℕ := 45
def science_homework_time : ℕ := 50
def history_homework_time : ℕ := 25
def special_project_time : ℕ := 30

-- Define the function to compute english homework time
def english_homework_time : ℕ :=
  total_time - (math_homework_time + science_homework_time + history_homework_time + special_project_time)

-- The theorem to show the English homework time is 30 minutes
theorem english_homework_correct_time :
  english_homework_time = 30 :=
  by
    sorry

end NUMINAMATH_GPT_english_homework_correct_time_l1448_144822


namespace NUMINAMATH_GPT_magnitude_of_T_l1448_144813

def i : Complex := Complex.I

def T : Complex := (1 + i) ^ 18 - (1 - i) ^ 18

theorem magnitude_of_T : Complex.abs T = 1024 := by
  sorry

end NUMINAMATH_GPT_magnitude_of_T_l1448_144813


namespace NUMINAMATH_GPT_khalil_paid_correct_amount_l1448_144814

-- Defining the charges for dogs and cats
def cost_per_dog : ℕ := 60
def cost_per_cat : ℕ := 40

-- Defining the number of dogs and cats Khalil took to the clinic
def num_dogs : ℕ := 20
def num_cats : ℕ := 60

-- The total amount Khalil paid
def total_amount_paid : ℕ := 3600

-- The theorem to prove the total amount Khalil paid
theorem khalil_paid_correct_amount :
  (cost_per_dog * num_dogs + cost_per_cat * num_cats) = total_amount_paid :=
by
  sorry

end NUMINAMATH_GPT_khalil_paid_correct_amount_l1448_144814


namespace NUMINAMATH_GPT_find_side_length_l1448_144865

noncomputable def cos (x : ℝ) := Real.cos x

theorem find_side_length
  (A : ℝ) (c : ℝ) (b : ℝ) (a : ℝ)
  (hA : A = Real.pi / 3)
  (hc : c = Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 3) :
  a = 3 := 
sorry

end NUMINAMATH_GPT_find_side_length_l1448_144865


namespace NUMINAMATH_GPT_miner_distance_when_explosion_heard_l1448_144809

-- Distance function for the miner (in feet)
def miner_distance (t : ℕ) : ℕ := 30 * t

-- Distance function for the sound after the explosion (in feet)
def sound_distance (t : ℕ) : ℕ := 1100 * (t - 45)

theorem miner_distance_when_explosion_heard :
  ∃ t : ℕ, miner_distance t / 3 = 463 ∧ miner_distance t = sound_distance t :=
sorry

end NUMINAMATH_GPT_miner_distance_when_explosion_heard_l1448_144809


namespace NUMINAMATH_GPT_factor_theorem_solution_l1448_144824

theorem factor_theorem_solution (t : ℝ) :
  (x - t ∣ 3 * x^2 + 10 * x - 8) ↔ (t = 2 / 3 ∨ t = -4) :=
by
  sorry

end NUMINAMATH_GPT_factor_theorem_solution_l1448_144824


namespace NUMINAMATH_GPT_arithmetic_sequence_inequality_l1448_144877

noncomputable def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

noncomputable def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_inequality
  (a d : ℕ)
  (i j k l : ℕ)
  (hi : i ≤ j)
  (hj : j ≤ k)
  (hk : k ≤ l)
  (hij: i + l = j + k)
  : (arithmetic_seq a d i) * (arithmetic_seq a d l) ≤ (arithmetic_seq a d j) * (arithmetic_seq a d k) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_inequality_l1448_144877


namespace NUMINAMATH_GPT_zero_of_f_when_m_is_neg1_monotonicity_of_f_m_gt_neg1_l1448_144806

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  x - 1/x - 2 * m * Real.log x

theorem zero_of_f_when_m_is_neg1 : ∃ x > 0, f x (-1) = 0 :=
  by
    use 1
    sorry

theorem monotonicity_of_f_m_gt_neg1 (m : ℝ) (hm : m > -1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x m ≤ f y m) ∨
  (∃ a b : ℝ, 0 < a ∧ a < b ∧
    (∀ x : ℝ, 0 < x ∧ x < a → f x m ≤ f a m) ∧
    (∀ x : ℝ, a < x ∧ x < b → f a m ≥ f x m) ∧
    (∀ x : ℝ, b < x → f b m ≤ f x m)) :=
  by
    cases lt_or_le m 1 with
    | inl hlt =>
        left
        intros x y hx hy hxy
        sorry
    | inr hle =>
        right
        use m - Real.sqrt (m^2 - 1), m + Real.sqrt (m^2 - 1)
        sorry

end NUMINAMATH_GPT_zero_of_f_when_m_is_neg1_monotonicity_of_f_m_gt_neg1_l1448_144806


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l1448_144884

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (c : ℝ)
  (hc : c^2 = a^2 + b^2) : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem eccentricity_of_hyperbola (a b c e : ℝ)
  (ha : a > 0) (hb : b > 0) (h_hyperbola : c^2 = a^2 + b^2)
  (h_eccentricity : e = (1 + Real.sqrt 5) / 2) :
  e = hyperbola_eccentricity a b ha hb c h_hyperbola :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l1448_144884


namespace NUMINAMATH_GPT_similar_triangles_legs_sum_l1448_144879

theorem similar_triangles_legs_sum (a b : ℕ) (h1 : a * b = 18) (h2 : a^2 + b^2 = 25) (bigger_area : ℕ) (smaller_area : ℕ) (hypotenuse : ℕ) 
  (h_similar : bigger_area = 225) 
  (h_smaller_area : smaller_area = 9) 
  (h_hypotenuse : hypotenuse = 5) 
  (h_non_3_4_5 : ¬ (a = 3 ∧ b = 4 ∨ a = 4 ∧ b = 3)) : 
  5 * (a + b) = 45 := 
by sorry

end NUMINAMATH_GPT_similar_triangles_legs_sum_l1448_144879


namespace NUMINAMATH_GPT_volume_of_rectangular_parallelepiped_l1448_144804

theorem volume_of_rectangular_parallelepiped (x y z p q r : ℝ) 
  (h1 : p = x * y) 
  (h2 : q = x * z) 
  (h3 : r = y * z) : 
  x * y * z = Real.sqrt (p * q * r) :=
by
  sorry

end NUMINAMATH_GPT_volume_of_rectangular_parallelepiped_l1448_144804


namespace NUMINAMATH_GPT_inequality_satisfaction_l1448_144898

theorem inequality_satisfaction (x y : ℝ) : 
  y - x < Real.sqrt (x^2) ↔ (y < 0 ∨ y < 2 * x) := by 
sorry

end NUMINAMATH_GPT_inequality_satisfaction_l1448_144898


namespace NUMINAMATH_GPT_race_distance_l1448_144830

theorem race_distance (D : ℝ) (h1 : ∀ t : ℝ, t = 30 → D / 30 = D / t)
                      (h2 : ∀ t : ℝ, t = 45 → D / 45 = D / t)
                      (h3 : ∀ d : ℝ, d = 33.333333333333336 → D - (D / 45) * 30 = d) :
  D = 100 :=
sorry

end NUMINAMATH_GPT_race_distance_l1448_144830


namespace NUMINAMATH_GPT_proper_sampling_method_l1448_144883

-- Definitions for conditions
def large_bulbs : ℕ := 120
def medium_bulbs : ℕ := 60
def small_bulbs : ℕ := 20
def sample_size : ℕ := 25

-- Definition for the proper sampling method to use
def sampling_method : String := "Stratified sampling"

-- Theorem statement to prove the sampling method
theorem proper_sampling_method :
  ∃ method : String, 
  method = sampling_method ∧
  sampling_method = "Stratified sampling" := by
    sorry

end NUMINAMATH_GPT_proper_sampling_method_l1448_144883


namespace NUMINAMATH_GPT_exists_even_function_b_l1448_144852

-- Define the function f(x) = 2x^2 - b*x
def f (b x : ℝ) : ℝ := 2 * x^2 - b * x

-- Define the condition for f being an even function: f(-x) = f(x)
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- The main theorem stating the existence of a b in ℝ such that f is an even function
theorem exists_even_function_b :
  ∃ b : ℝ, is_even_function (f b) :=
by
  sorry

end NUMINAMATH_GPT_exists_even_function_b_l1448_144852
