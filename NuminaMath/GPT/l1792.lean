import Mathlib

namespace NUMINAMATH_GPT_compute_k_plus_m_l1792_179278

theorem compute_k_plus_m :
  ∃ k m : ℝ, 
    (∀ (x y z : ℝ), x^3 - 9 * x^2 + k * x - m = 0 -> x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 9 ∧ 
    (x = 1 ∨ y = 1 ∨ z = 1) ∧ (x = 3 ∨ y = 3 ∨ z = 3) ∧ (x = 5 ∨ y = 5 ∨ z = 5)) →
    k + m = 38 :=
by
  sorry

end NUMINAMATH_GPT_compute_k_plus_m_l1792_179278


namespace NUMINAMATH_GPT_calories_for_breakfast_l1792_179246

theorem calories_for_breakfast :
  let cake_calories := 110
  let chips_calories := 310
  let coke_calories := 215
  let lunch_calories := 780
  let daily_limit := 2500
  let remaining_calories := 525
  let total_dinner_snacks := cake_calories + chips_calories + coke_calories
  let total_lunch_dinner := total_dinner_snacks + lunch_calories
  let total_consumed := daily_limit - remaining_calories
  total_consumed - total_lunch_dinner = 560 := by
  sorry

end NUMINAMATH_GPT_calories_for_breakfast_l1792_179246


namespace NUMINAMATH_GPT_number_of_smaller_cubes_l1792_179214

theorem number_of_smaller_cubes (N : ℕ) : 
  (∀ a : ℕ, ∃ n : ℕ, n * a^3 = 125) ∧
  (∀ b : ℕ, b ≤ 5 → ∃ m : ℕ, m * b^3 ≤ 125) ∧
  (∃ x y : ℕ, x ≠ y) → 
  N = 118 :=
sorry

end NUMINAMATH_GPT_number_of_smaller_cubes_l1792_179214


namespace NUMINAMATH_GPT_time_spent_on_aerobics_l1792_179258

theorem time_spent_on_aerobics (A W : ℝ) 
  (h1 : A + W = 250) 
  (h2 : A / W = 3 / 2) : 
  A = 150 := 
sorry

end NUMINAMATH_GPT_time_spent_on_aerobics_l1792_179258


namespace NUMINAMATH_GPT_least_sugar_l1792_179222

theorem least_sugar (f s : ℚ) (h1 : f ≥ 10 + 3 * s / 4) (h2 : f ≤ 3 * s) :
  s ≥ 40 / 9 :=
  sorry

end NUMINAMATH_GPT_least_sugar_l1792_179222


namespace NUMINAMATH_GPT_divisor_in_first_division_l1792_179203

theorem divisor_in_first_division
  (N : ℕ)
  (D : ℕ)
  (Q : ℕ)
  (h1 : N = 8 * D)
  (h2 : N % 5 = 4) :
  D = 3 := 
sorry

end NUMINAMATH_GPT_divisor_in_first_division_l1792_179203


namespace NUMINAMATH_GPT_company_production_average_l1792_179291

theorem company_production_average (n : ℕ) 
  (h1 : (50 * n) / n = 50) 
  (h2 : (50 * n + 105) / (n + 1) = 55) :
  n = 10 :=
sorry

end NUMINAMATH_GPT_company_production_average_l1792_179291


namespace NUMINAMATH_GPT_problem1_problem2_min_value_l1792_179289

theorem problem1 (x : ℝ) : |x + 1| + |x - 2| ≥ 3 := sorry

theorem problem2 (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 14 := sorry

theorem min_value (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) :
  ∃ x y z, x^2 + y^2 + z^2 = 1 / 14 := sorry

end NUMINAMATH_GPT_problem1_problem2_min_value_l1792_179289


namespace NUMINAMATH_GPT_sum_infinite_series_l1792_179284

theorem sum_infinite_series : (∑' n : ℕ, (n + 1) / 8^(n + 1)) = 8 / 49 := sorry

end NUMINAMATH_GPT_sum_infinite_series_l1792_179284


namespace NUMINAMATH_GPT_marcy_total_people_served_l1792_179243

noncomputable def total_people_served_lip_gloss
  (tubs_lip_gloss : ℕ) (tubes_per_tub_lip_gloss : ℕ) (people_per_tube_lip_gloss : ℕ) : ℕ :=
  tubs_lip_gloss * tubes_per_tub_lip_gloss * people_per_tube_lip_gloss

noncomputable def total_people_served_mascara
  (tubs_mascara : ℕ) (tubes_per_tub_mascara : ℕ) (people_per_tube_mascara : ℕ) : ℕ :=
  tubs_mascara * tubes_per_tub_mascara * people_per_tube_mascara

theorem marcy_total_people_served :
  ∀ (tubs_lip_gloss tubs_mascara : ℕ) 
    (tubes_per_tub_lip_gloss tubes_per_tub_mascara 
     people_per_tube_lip_gloss people_per_tube_mascara : ℕ),
    tubs_lip_gloss = 6 → 
    tubes_per_tub_lip_gloss = 2 → 
    people_per_tube_lip_gloss = 3 → 
    tubs_mascara = 4 → 
    tubes_per_tub_mascara = 3 → 
    people_per_tube_mascara = 5 → 
    total_people_served_lip_gloss tubs_lip_gloss 
                                 tubes_per_tub_lip_gloss 
                                 people_per_tube_lip_gloss = 36 :=
by
  intros tubs_lip_gloss tubs_mascara 
         tubes_per_tub_lip_gloss tubes_per_tub_mascara 
         people_per_tube_lip_gloss people_per_tube_mascara
         h_tubs_lip_gloss h_tubes_per_tub_lip_gloss h_people_per_tube_lip_gloss
         h_tubs_mascara h_tubes_per_tub_mascara h_people_per_tube_mascara
  rw [h_tubs_lip_gloss, h_tubes_per_tub_lip_gloss, h_people_per_tube_lip_gloss]
  exact rfl


end NUMINAMATH_GPT_marcy_total_people_served_l1792_179243


namespace NUMINAMATH_GPT_cubic_meter_to_cubic_centimeters_l1792_179264

theorem cubic_meter_to_cubic_centimeters : 
  (1 : ℝ)^3 = (100 : ℝ)^3 * (1 : ℝ)^0 := 
by 
  sorry

end NUMINAMATH_GPT_cubic_meter_to_cubic_centimeters_l1792_179264


namespace NUMINAMATH_GPT_maximum_unique_numbers_in_circle_l1792_179212

theorem maximum_unique_numbers_in_circle :
  ∀ (n : ℕ) (numbers : ℕ → ℤ), n = 2023 →
  (∀ i, numbers i = numbers ((i + 1) % n) * numbers ((i + n - 1) % n)) →
  ∀ i j, numbers i = numbers j :=
by
  sorry

end NUMINAMATH_GPT_maximum_unique_numbers_in_circle_l1792_179212


namespace NUMINAMATH_GPT_mapping_has_output_l1792_179219

variable (M N : Type) (f : M → N)

theorem mapping_has_output (x : M) : ∃ y : N, f x = y :=
by
  sorry

end NUMINAMATH_GPT_mapping_has_output_l1792_179219


namespace NUMINAMATH_GPT_percentage_increase_after_decrease_and_increase_l1792_179238

theorem percentage_increase_after_decrease_and_increase 
  (P : ℝ) 
  (h : 0.8 * P + (x / 100) * (0.8 * P) = 1.16 * P) : 
  x = 45 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_after_decrease_and_increase_l1792_179238


namespace NUMINAMATH_GPT_average_speed_palindrome_l1792_179225

theorem average_speed_palindrome :
  ∀ (initial_odometer final_odometer : ℕ) (hours : ℕ),
  initial_odometer = 123321 →
  final_odometer = 124421 →
  hours = 4 →
  (final_odometer - initial_odometer) / hours = 275 :=
by
  intros initial_odometer final_odometer hours h1 h2 h3
  sorry

end NUMINAMATH_GPT_average_speed_palindrome_l1792_179225


namespace NUMINAMATH_GPT_integer_expression_l1792_179223

theorem integer_expression (m : ℤ) : ∃ k : ℤ, k = (m / 3) + (m^2 / 2) + (m^3 / 6) :=
sorry

end NUMINAMATH_GPT_integer_expression_l1792_179223


namespace NUMINAMATH_GPT_real_part_of_diff_times_i_l1792_179248

open Complex

def z1 : ℂ := (4 : ℂ) + (29 : ℂ) * I
def z2 : ℂ := (6 : ℂ) + (9 : ℂ) * I

theorem real_part_of_diff_times_i :
  re ((z1 - z2) * I) = -20 := 
sorry

end NUMINAMATH_GPT_real_part_of_diff_times_i_l1792_179248


namespace NUMINAMATH_GPT_numberOfFlowerbeds_l1792_179292

def totalSeeds : ℕ := 32
def seedsPerFlowerbed : ℕ := 4

theorem numberOfFlowerbeds : totalSeeds / seedsPerFlowerbed = 8 :=
by
  sorry

end NUMINAMATH_GPT_numberOfFlowerbeds_l1792_179292


namespace NUMINAMATH_GPT_minimize_cylinder_surface_area_l1792_179265

noncomputable def cylinder_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem minimize_cylinder_surface_area :
  ∃ r h : ℝ, cylinder_volume r h = 16 * Real.pi ∧
  (∀ r' h', cylinder_volume r' h' = 16 * Real.pi → cylinder_surface_area r h ≤ cylinder_surface_area r' h') ∧ r = 2 := by
  sorry

end NUMINAMATH_GPT_minimize_cylinder_surface_area_l1792_179265


namespace NUMINAMATH_GPT_incircle_area_of_triangle_l1792_179231

noncomputable def hyperbola_params : Type :=
  sorry

noncomputable def point_on_hyperbola (P : hyperbola_params) : Prop :=
  sorry

noncomputable def in_first_quadrant (P : hyperbola_params) : Prop :=
  sorry

noncomputable def distance_ratio (PF1 PF2 : ℝ) : Prop :=
  PF1 / PF2 = 4 / 3

noncomputable def distance1_is_8 (PF1 : ℝ) : Prop :=
  PF1 = 8

noncomputable def distance2_is_6 (PF2 : ℝ) : Prop :=
  PF2 = 6

noncomputable def distance_between_foci (F1F2 : ℝ) : Prop :=
  F1F2 = 10

noncomputable def incircle_area (area : ℝ) : Prop :=
  area = 4 * Real.pi

theorem incircle_area_of_triangle (P : hyperbola_params) 
  (hP : point_on_hyperbola P) 
  (h1 : in_first_quadrant P)
  (PF1 PF2 : ℝ)
  (h2 : distance_ratio PF1 PF2)
  (h3 : distance1_is_8 PF1)
  (h4 : distance2_is_6 PF2)
  (F1F2 : ℝ) 
  (h5 : distance_between_foci F1F2) :
  ∃ r : ℝ, incircle_area (Real.pi * r^2) :=
by
  sorry

end NUMINAMATH_GPT_incircle_area_of_triangle_l1792_179231


namespace NUMINAMATH_GPT_money_left_l1792_179220

theorem money_left 
  (salary : ℝ)
  (spent_on_food : ℝ)
  (spent_on_rent : ℝ)
  (spent_on_clothes : ℝ)
  (total_spent : ℝ)
  (money_left : ℝ)
  (h_salary : salary = 170000)
  (h_food : spent_on_food = salary * (1 / 5))
  (h_rent : spent_on_rent = salary * (1 / 10))
  (h_clothes : spent_on_clothes = salary * (3 / 5))
  (h_total_spent : total_spent = spent_on_food + spent_on_rent + spent_on_clothes)
  (h_money_left : money_left = salary - total_spent) :
  money_left = 17000 :=
by
  sorry

end NUMINAMATH_GPT_money_left_l1792_179220


namespace NUMINAMATH_GPT_sum_of_digits_M_l1792_179213

-- Definitions
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Conditions
variables (M : ℕ)
  (h1 : M % 2 = 0)  -- M is even
  (h2 : ∀ d ∈ M.digits 10, d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 5 ∨ d = 7 ∨ d = 9)  -- Digits of M
  (h3 : sum_of_digits (2 * M) = 31)  -- Sum of digits of 2M
  (h4 : sum_of_digits (M / 2) = 28)  -- Sum of digits of M/2

-- Goal
theorem sum_of_digits_M :
  sum_of_digits M = 29 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_M_l1792_179213


namespace NUMINAMATH_GPT_valid_integer_values_of_x_l1792_179207

theorem valid_integer_values_of_x (x : ℤ) 
  (h1 : 3 < x) (h2 : x < 10)
  (h3 : 5 < x) (h4 : x < 18)
  (h5 : -2 < x) (h6 : x < 9)
  (h7 : 0 < x) (h8 : x < 8) 
  (h9 : x + 1 < 9) : x = 6 ∨ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_valid_integer_values_of_x_l1792_179207


namespace NUMINAMATH_GPT_polar_bear_daily_fish_intake_l1792_179297

theorem polar_bear_daily_fish_intake : 
  (0.2 + 0.4 = 0.6) := by
  sorry

end NUMINAMATH_GPT_polar_bear_daily_fish_intake_l1792_179297


namespace NUMINAMATH_GPT_mary_should_drink_6_glasses_l1792_179209

-- Definitions based on conditions
def daily_water_goal_liters : ℚ := 1.5
def glass_capacity_ml : ℚ := 250
def liter_to_milliliters : ℚ := 1000

-- Conversion from liters to milliliters
def daily_water_goal_milliliters : ℚ := daily_water_goal_liters * liter_to_milliliters

-- Proof problem to show Mary needs 6 glasses per day
theorem mary_should_drink_6_glasses :
  daily_water_goal_milliliters / glass_capacity_ml = 6 := by
  sorry

end NUMINAMATH_GPT_mary_should_drink_6_glasses_l1792_179209


namespace NUMINAMATH_GPT_sum_of_ages_in_three_years_l1792_179273

theorem sum_of_ages_in_three_years (H : ℕ) (J : ℕ) (SumAges : ℕ) 
  (h1 : J = 3 * H) 
  (h2 : H = 15) 
  (h3 : SumAges = (H + 3) + (J + 3)) : 
  SumAges = 66 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_in_three_years_l1792_179273


namespace NUMINAMATH_GPT_peter_invested_for_3_years_l1792_179218

-- Definitions of parameters
def P : ℝ := 650
def APeter : ℝ := 815
def ADavid : ℝ := 870
def tDavid : ℝ := 4

-- Simple interest formula for Peter
def simple_interest_peter (r : ℝ) (t : ℝ) : Prop :=
  APeter = P + P * r * t

-- Simple interest formula for David
def simple_interest_david (r : ℝ) : Prop :=
  ADavid = P + P * r * tDavid

-- The main theorem to find out how many years Peter invested his money
theorem peter_invested_for_3_years : ∃ t : ℝ, (∃ r : ℝ, simple_interest_peter r t ∧ simple_interest_david r) ∧ t = 3 :=
by
  sorry

end NUMINAMATH_GPT_peter_invested_for_3_years_l1792_179218


namespace NUMINAMATH_GPT_polynomial_coeffs_l1792_179256

theorem polynomial_coeffs :
  ( ∃ (a1 a2 a3 a4 a5 : ℕ), (∀ (x : ℝ), (x + 1) ^ 3 * (x + 2) ^ 2 = x^5 + a1 * x^4 + a2 * x^3 + a3 * x^2 + a4 * x + a5) ∧ a4 = 16 ∧ a5 = 4) := 
by
  sorry

end NUMINAMATH_GPT_polynomial_coeffs_l1792_179256


namespace NUMINAMATH_GPT_xy_fraction_l1792_179295

theorem xy_fraction (x y : ℚ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) :
  x * y = -1 / 5 := 
by sorry

end NUMINAMATH_GPT_xy_fraction_l1792_179295


namespace NUMINAMATH_GPT_angle_A_value_sin_2B_plus_A_l1792_179226

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (h1 : a = 3)
variable (h2 : b = 2 * Real.sqrt 2)
variable (triangle_condition : b / (a + c) = 1 - (Real.sin C / (Real.sin A + Real.sin B)))

theorem angle_A_value : A = Real.pi / 3 :=
sorry

theorem sin_2B_plus_A (hA : A = Real.pi / 3) : 
  Real.sin (2 * B + A) = (2 * Real.sqrt 2 - Real.sqrt 3) / 6 :=
sorry

end NUMINAMATH_GPT_angle_A_value_sin_2B_plus_A_l1792_179226


namespace NUMINAMATH_GPT_P_subset_Q_l1792_179281

def P (x : ℝ) := abs x < 2
def Q (x : ℝ) := x < 2

theorem P_subset_Q : ∀ x : ℝ, P x → Q x := by
  sorry

end NUMINAMATH_GPT_P_subset_Q_l1792_179281


namespace NUMINAMATH_GPT_find_m_l1792_179232

variables (a b : ℝ × ℝ) (m : ℝ)

def vectors := (a = (3, 4)) ∧ (b = (2, -1))

def perpendicular (a b : ℝ × ℝ) : Prop :=
a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (h1 : vectors a b) (h2 : perpendicular (a.1 + m * b.1, a.2 + m * b.2) (a.1 - b.1, a.2 - b.2)) :
  m = 23 / 3 :=
sorry

end NUMINAMATH_GPT_find_m_l1792_179232


namespace NUMINAMATH_GPT_nancy_deleted_files_correct_l1792_179282

-- Variables and conditions
def nancy_original_files : Nat := 43
def files_per_folder : Nat := 6
def number_of_folders : Nat := 2

-- Definition of the number of files that were deleted
def nancy_files_deleted : Nat :=
  nancy_original_files - (files_per_folder * number_of_folders)

-- Theorem to prove
theorem nancy_deleted_files_correct :
  nancy_files_deleted = 31 :=
by
  sorry

end NUMINAMATH_GPT_nancy_deleted_files_correct_l1792_179282


namespace NUMINAMATH_GPT_tetrahedron_edge_assignment_possible_l1792_179276

theorem tetrahedron_edge_assignment_possible 
(s S a b : ℝ) 
(hs : s ≥ 0) (hS : S ≥ 0) (ha : a ≥ 0) (hb : b ≥ 0) :
  ∃ (e₁ e₂ e₃ e₄ e₅ e₆ : ℝ),
    e₁ ≥ 0 ∧ e₂ ≥ 0 ∧ e₃ ≥ 0 ∧ e₄ ≥ 0 ∧ e₅ ≥ 0 ∧ e₆ ≥ 0 ∧
    (e₁ + e₂ + e₃ = s) ∧ (e₁ + e₄ + e₅ = S) ∧
    (e₂ + e₄ + e₆ = a) ∧ (e₃ + e₅ + e₆ = b) := by
  sorry

end NUMINAMATH_GPT_tetrahedron_edge_assignment_possible_l1792_179276


namespace NUMINAMATH_GPT_Raja_and_Ram_together_l1792_179293

def RajaDays : ℕ := 12
def RamDays : ℕ := 6

theorem Raja_and_Ram_together (W : ℕ) : 
  let RajaRate := W / RajaDays
  let RamRate := W / RamDays
  let CombinedRate := RajaRate + RamRate 
  let DaysTogether := W / CombinedRate 
  DaysTogether = 4 := 
by
  sorry

end NUMINAMATH_GPT_Raja_and_Ram_together_l1792_179293


namespace NUMINAMATH_GPT_train_length_l1792_179244

noncomputable def length_of_each_train : ℝ :=
  let speed_faster_train_km_per_hr := 46
  let speed_slower_train_km_per_hr := 36
  let relative_speed_km_per_hr := speed_faster_train_km_per_hr - speed_slower_train_km_per_hr
  let relative_speed_m_per_s := (relative_speed_km_per_hr * 1000) / 3600
  let time_s := 54
  let distance_m := relative_speed_m_per_s * time_s
  distance_m / 2

theorem train_length : length_of_each_train = 75 := by
  sorry

end NUMINAMATH_GPT_train_length_l1792_179244


namespace NUMINAMATH_GPT_smallest_twice_perfect_square_three_times_perfect_cube_l1792_179205

theorem smallest_twice_perfect_square_three_times_perfect_cube :
  ∃ n : ℕ, (∃ k : ℕ, n = 2 * k^2) ∧ (∃ m : ℕ, n = 3 * m^3) ∧ n = 648 :=
by
  sorry

end NUMINAMATH_GPT_smallest_twice_perfect_square_three_times_perfect_cube_l1792_179205


namespace NUMINAMATH_GPT_bacteria_growth_rate_l1792_179259

-- Define the existence of the growth rate and the initial amount of bacteria
variable (B : ℕ → ℝ) (B0 : ℝ) (r : ℝ)

-- State the conditions from the problem
axiom bacteria_growth_model : ∀ t : ℕ, B t = B0 * r ^ t
axiom day_30_full : B 30 = B0 * r ^ 30
axiom day_26_sixteenth : B 26 = (1 / 16) * B 30

-- Theorem stating that the growth rate r of the bacteria each day is 2
theorem bacteria_growth_rate : r = 2 := by
  sorry

end NUMINAMATH_GPT_bacteria_growth_rate_l1792_179259


namespace NUMINAMATH_GPT_probability_yellow_or_blue_twice_l1792_179254

theorem probability_yellow_or_blue_twice :
  let total_faces := 12
  let yellow_faces := 4
  let blue_faces := 2
  let probability_yellow_or_blue := (yellow_faces / total_faces) + (blue_faces / total_faces)
  (probability_yellow_or_blue * probability_yellow_or_blue) = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_probability_yellow_or_blue_twice_l1792_179254


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l1792_179294

theorem area_of_triangle_ABC 
  (ABCD_is_trapezoid : ∀ {a b c d : ℝ}, a + d = b + c)
  (area_ABCD : ∀ {a b : ℝ}, a * b = 24)
  (CD_three_times_AB : ∀ {a : ℝ}, a * 3 = 24) :
  ∃ (area_ABC : ℝ), area_ABC = 6 :=
by 
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l1792_179294


namespace NUMINAMATH_GPT_find_number_l1792_179252

theorem find_number (x : ℝ) (h : x / 0.04 = 25) : x = 1 := 
by 
  -- the steps for solving this will be provided here
  sorry

end NUMINAMATH_GPT_find_number_l1792_179252


namespace NUMINAMATH_GPT_arithmetic_sequence_difference_l1792_179285

theorem arithmetic_sequence_difference (a b c : ℤ) (d : ℤ)
  (h1 : 9 - 1 = 4 * d)
  (h2 : c - a = 2 * d) :
  c - a = 4 := by sorry

end NUMINAMATH_GPT_arithmetic_sequence_difference_l1792_179285


namespace NUMINAMATH_GPT_sale_first_month_l1792_179200

-- Declaration of all constant sales amounts in rupees
def sale_second_month : ℕ := 6927
def sale_third_month : ℕ := 6855
def sale_fourth_month : ℕ := 7230
def sale_fifth_month : ℕ := 6562
def sale_sixth_month : ℕ := 6791
def average_required : ℕ := 6800
def months : ℕ := 6

-- Total sales computed from the average sale requirement
def total_sales_needed : ℕ := months * average_required

-- The sum of sales for the second to sixth months
def total_sales_last_five_months := sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month

-- Prove the sales in the first month given the conditions
theorem sale_first_month :
  total_sales_needed - total_sales_last_five_months = 6435 :=
by
  sorry

end NUMINAMATH_GPT_sale_first_month_l1792_179200


namespace NUMINAMATH_GPT_koi_fish_in_pond_l1792_179204

theorem koi_fish_in_pond:
  ∃ k : ℕ, 2 * k - 14 = 64 ∧ k = 39 := sorry

end NUMINAMATH_GPT_koi_fish_in_pond_l1792_179204


namespace NUMINAMATH_GPT_right_triangle_area_l1792_179235

theorem right_triangle_area (a b c : ℝ) (h : c = 5) (h1 : a = 3) (h2 : c^2 = a^2 + b^2) : 
  1 / 2 * a * b = 6 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1792_179235


namespace NUMINAMATH_GPT_soap_bubble_thickness_scientific_notation_l1792_179236

theorem soap_bubble_thickness_scientific_notation :
  (0.0007 * 0.001) = 7 * 10^(-7) := by
sorry

end NUMINAMATH_GPT_soap_bubble_thickness_scientific_notation_l1792_179236


namespace NUMINAMATH_GPT_range_of_mu_l1792_179275

theorem range_of_mu (a b μ : ℝ) (ha : 0 < a) (hb : 0 < b) (hμ : 0 < μ) (h : 1 / a + 9 / b = 1) : μ ≤ 16 :=
by
  sorry

end NUMINAMATH_GPT_range_of_mu_l1792_179275


namespace NUMINAMATH_GPT_root_expression_eq_l1792_179288

theorem root_expression_eq (p q α β γ δ : ℝ) 
  (h1 : ∀ x, (x - α) * (x - β) = x^2 + p * x + 2)
  (h2 : ∀ x, (x - γ) * (x - δ) = x^2 + q * x + 2) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = 4 + 2 * (p^2 - q^2) := 
sorry

end NUMINAMATH_GPT_root_expression_eq_l1792_179288


namespace NUMINAMATH_GPT_maximum_figures_per_shelf_l1792_179266

theorem maximum_figures_per_shelf
  (figures_shelf_1 : ℕ)
  (figures_shelf_2 : ℕ)
  (figures_shelf_3 : ℕ)
  (additional_shelves : ℕ)
  (max_figures_per_shelf : ℕ)
  (total_figures : ℕ)
  (total_shelves : ℕ)
  (H1 : figures_shelf_1 = 9)
  (H2 : figures_shelf_2 = 14)
  (H3 : figures_shelf_3 = 7)
  (H4 : additional_shelves = 2)
  (H5 : max_figures_per_shelf = 11)
  (H6 : total_figures = figures_shelf_1 + figures_shelf_2 + figures_shelf_3)
  (H7 : total_shelves = 3 + additional_shelves)
  (H8 : ∃ d, d ∈ ({x : ℕ | x ∣ total_figures} ∩ {y : ℕ | y ≤ max_figures_per_shelf}))
  : ∃ d, d ∈ ({x : ℕ | x ∣ total_figures} ∩ {y : ℕ | y ≤ max_figures_per_shelf}) ∧ d = 6 := sorry

end NUMINAMATH_GPT_maximum_figures_per_shelf_l1792_179266


namespace NUMINAMATH_GPT_margaret_mean_score_l1792_179228

def sum_of_scores (scores : List ℤ) : ℤ :=
  scores.sum

def mean_score (total_score : ℤ) (count : ℕ) : ℚ :=
  total_score / count

theorem margaret_mean_score :
  let scores := [85, 88, 90, 92, 94, 96, 100]
  let cyprian_mean := 92
  let cyprian_count := 4
  let total_score := sum_of_scores scores
  let cyprian_total_score := cyprian_mean * cyprian_count
  let margaret_total_score := total_score - cyprian_total_score
  let margaret_mean := mean_score margaret_total_score 3
  margaret_mean = 92.33 :=
by
  sorry

end NUMINAMATH_GPT_margaret_mean_score_l1792_179228


namespace NUMINAMATH_GPT_cubes_sum_correct_l1792_179251

noncomputable def max_cubes : ℕ := 11
noncomputable def min_cubes : ℕ := 9

theorem cubes_sum_correct : max_cubes + min_cubes = 20 :=
by
  unfold max_cubes min_cubes
  sorry

end NUMINAMATH_GPT_cubes_sum_correct_l1792_179251


namespace NUMINAMATH_GPT_division_theorem_l1792_179208

variable (x : ℤ)

def dividend := 8 * x ^ 4 + 7 * x ^ 3 + 3 * x ^ 2 - 5 * x - 8
def divisor := x - 1
def quotient := 8 * x ^ 3 + 15 * x ^ 2 + 18 * x + 13
def remainder := 5

theorem division_theorem : dividend x = divisor x * quotient x + remainder := by
  sorry

end NUMINAMATH_GPT_division_theorem_l1792_179208


namespace NUMINAMATH_GPT_BowlingAlleyTotalPeople_l1792_179239

/--
There are 31 groups of people at the bowling alley.
Each group has about 6 people.
Prove that the total number of people at the bowling alley is 186.
-/
theorem BowlingAlleyTotalPeople : 
  let groups := 31
  let people_per_group := 6
  groups * people_per_group = 186 :=
by
  sorry

end NUMINAMATH_GPT_BowlingAlleyTotalPeople_l1792_179239


namespace NUMINAMATH_GPT_possible_values_of_a_l1792_179230

theorem possible_values_of_a (x y a : ℝ)
  (h1 : x + y = a)
  (h2 : x^3 + y^3 = a)
  (h3 : x^5 + y^5 = a) :
  a = 0 ∨ a = 1 ∨ a = -1 ∨ a = 2 ∨ a = -2 :=
sorry

end NUMINAMATH_GPT_possible_values_of_a_l1792_179230


namespace NUMINAMATH_GPT_shift_line_one_unit_left_l1792_179255

theorem shift_line_one_unit_left : ∀ (x y : ℝ), (y = x) → (y - 1 = (x + 1) - 1) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_shift_line_one_unit_left_l1792_179255


namespace NUMINAMATH_GPT_problem_l1792_179262

theorem problem (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ q) : ¬ p ∧ q :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_problem_l1792_179262


namespace NUMINAMATH_GPT_probability_two_points_one_unit_apart_l1792_179272

def twelve_points_probability : ℚ := 2 / 11

/-- Twelve points are spaced around at intervals of one unit around a \(3 \times 3\) square.
    Two of the 12 points are chosen at random.
    Prove that the probability that the two points are one unit apart is \(\frac{2}{11}\). -/
theorem probability_two_points_one_unit_apart :
  let total_points := 12
  let total_combinations := (total_points * (total_points - 1)) / 2
  let favorable_pairs := 12
  (favorable_pairs : ℚ) / total_combinations = twelve_points_probability := by
  sorry

end NUMINAMATH_GPT_probability_two_points_one_unit_apart_l1792_179272


namespace NUMINAMATH_GPT_mapping_sum_l1792_179253

theorem mapping_sum (f : ℝ × ℝ → ℝ × ℝ) (a b : ℝ)
(h1 : ∀ x y, f (x, y) = (x, x + y))
(h2 : (a, b) = f (1, 3)) :
  a + b = 5 :=
sorry

end NUMINAMATH_GPT_mapping_sum_l1792_179253


namespace NUMINAMATH_GPT_determinant_2x2_l1792_179268

theorem determinant_2x2 (a b c d : ℝ) 
  (h : Matrix.det (Matrix.of ![![1, a, b], ![2, c, d], ![3, 0, 0]]) = 6) : 
  Matrix.det (Matrix.of ![![a, b], ![c, d]]) = 2 :=
by
  sorry

end NUMINAMATH_GPT_determinant_2x2_l1792_179268


namespace NUMINAMATH_GPT_range_of_m_l1792_179229

theorem range_of_m (m : ℝ) (x : ℝ) (hp : (x + 2) * (x - 10) ≤ 0)
  (hq : x^2 - 2 * x + 1 - m^2 ≤ 0) (hm : m > 0) : 0 < m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1792_179229


namespace NUMINAMATH_GPT_simplify_expression_correct_l1792_179263

noncomputable def simplify_expr (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :=
  ((a / b) * ((b - (4 * (a^6) / b^3)) ^ (1 / 3))
    - a^2 * ((b / a^6 - (4 / b^3)) ^ (1 / 3))
    + (2 / (a * b)) * ((a^3 * b^4 - 4 * a^9) ^ (1 / 3))) /
    ((b^2 - 2 * a^3) ^ (1 / 3) / b^2)

theorem simplify_expression_correct (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  simplify_expr a b ha hb = (a + b) * ((b^2 + 2 * a^3) ^ (1 / 3)) :=
sorry

end NUMINAMATH_GPT_simplify_expression_correct_l1792_179263


namespace NUMINAMATH_GPT_find_first_number_l1792_179257

theorem find_first_number (x : ℝ) : (10 + 70 + 28) / 3 = 36 →
  (x + 40 + 60) / 3 = 40 →
  x = 20 := 
by
  intros h_avg_old h_avg_new
  sorry

end NUMINAMATH_GPT_find_first_number_l1792_179257


namespace NUMINAMATH_GPT_hockey_season_games_l1792_179267

theorem hockey_season_games (n_teams : ℕ) (n_faces : ℕ) (h1 : n_teams = 18) (h2 : n_faces = 10) :
  let total_games := (n_teams * (n_teams - 1) / 2) * n_faces
  total_games = 1530 :=
by
  sorry

end NUMINAMATH_GPT_hockey_season_games_l1792_179267


namespace NUMINAMATH_GPT_number_division_equals_value_l1792_179274

theorem number_division_equals_value (x : ℝ) (h : x / 0.144 = 14.4 / 0.0144) : x = 144 :=
by
  sorry

end NUMINAMATH_GPT_number_division_equals_value_l1792_179274


namespace NUMINAMATH_GPT_initial_customers_l1792_179260

theorem initial_customers (x : ℝ) : (x - 8 + 4 = 9) → x = 13 :=
by
  sorry

end NUMINAMATH_GPT_initial_customers_l1792_179260


namespace NUMINAMATH_GPT_apples_bought_l1792_179287

theorem apples_bought (x : ℕ) 
  (h1 : x ≠ 0)  -- x must be a positive integer
  (h2 : 2 * (x/3) = 2 * x / 3 + 2 - 6) : x = 24 := 
  by sorry

end NUMINAMATH_GPT_apples_bought_l1792_179287


namespace NUMINAMATH_GPT_initial_books_in_bin_l1792_179261

theorem initial_books_in_bin
  (x : ℝ)
  (h : x + 33.0 + 2.0 = 76) :
  x = 41.0 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_initial_books_in_bin_l1792_179261


namespace NUMINAMATH_GPT_find_extrema_of_f_l1792_179271

noncomputable def f (x : ℝ) := x^2 - 4 * x - 2

theorem find_extrema_of_f : 
  (∀ x, (1 ≤ x ∧ x ≤ 4) → f x ≤ -2) ∧ 
  (∃ x, (1 ≤ x ∧ x ≤ 4 ∧ f x = -6)) :=
by sorry

end NUMINAMATH_GPT_find_extrema_of_f_l1792_179271


namespace NUMINAMATH_GPT_range_of_a_l1792_179279

theorem range_of_a (a : ℝ) : ((1 - a) ^ 2 + (1 + a) ^ 2 < 4) → (-1 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1792_179279


namespace NUMINAMATH_GPT_am_minus_gm_less_than_option_D_l1792_179211

variable (c d : ℝ)
variable (hc_pos : 0 < c) (hd_pos : 0 < d) (hcd_lt : c < d)

noncomputable def am : ℝ := (c + d) / 2
noncomputable def gm : ℝ := Real.sqrt (c * d)

theorem am_minus_gm_less_than_option_D :
  (am c d - gm c d) < ((d - c) ^ 3 / (8 * c)) :=
sorry

end NUMINAMATH_GPT_am_minus_gm_less_than_option_D_l1792_179211


namespace NUMINAMATH_GPT_proof_problem_l1792_179217

def p : Prop := ∃ x : ℝ, x^2 - x + 1 ≥ 0
def q : Prop := ∀ (a b : ℝ), (a^2 < b^2) → (a < b)

theorem proof_problem (h₁ : p) (h₂ : ¬ q) : p ∧ ¬ q := by
  exact ⟨h₁, h₂⟩

end NUMINAMATH_GPT_proof_problem_l1792_179217


namespace NUMINAMATH_GPT_xy_sum_value_l1792_179298

theorem xy_sum_value (x y : ℝ) (h1 : x^2 + x * y + 2 * y = 10) (h2 : y^2 + x * y + 2 * x = 14) :
  x + y = -6 ∨ x + y = 4 :=
sorry

end NUMINAMATH_GPT_xy_sum_value_l1792_179298


namespace NUMINAMATH_GPT_r_p_q_sum_l1792_179224

theorem r_p_q_sum (t p q r : ℕ) (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4)
    (h2 : (1 - Real.sin t) * (1 - Real.cos t) = p / q - Real.sqrt r)
    (h3 : r > 0) (h4 : p > 0) (h5 : q > 0)
    (h6 : Nat.gcd p q = 1) : r + p + q = 5 := 
sorry

end NUMINAMATH_GPT_r_p_q_sum_l1792_179224


namespace NUMINAMATH_GPT_gifts_wrapped_with_third_roll_l1792_179221

def num_rolls : ℕ := 3
def num_gifts : ℕ := 12
def first_roll_gifts : ℕ := 3
def second_roll_gifts : ℕ := 5

theorem gifts_wrapped_with_third_roll : 
  first_roll_gifts + second_roll_gifts < num_gifts → 
  num_gifts - (first_roll_gifts + second_roll_gifts) = 4 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_gifts_wrapped_with_third_roll_l1792_179221


namespace NUMINAMATH_GPT_rich_walked_distance_l1792_179242

def total_distance_walked (d1 d2 : ℕ) := 
  d1 + d2 + 2 * (d1 + d2) + (d1 + d2 + 2 * (d1 + d2)) / 2

def distance_to_intersection (d1 d2 : ℕ) := 
  2 * (d1 + d2)

def distance_to_end_route (d1 d2 : ℕ) := 
  (d1 + d2 + distance_to_intersection d1 d2) / 2

def total_distance_one_way (d1 d2 : ℕ) := 
  (d1 + d2) + (distance_to_intersection d1 d2) + (distance_to_end_route d1 d2)

theorem rich_walked_distance
  (d1 : ℕ := 20)
  (d2 : ℕ := 200) :
  2 * total_distance_one_way d1 d2 = 1980 :=
by
  simp [total_distance_one_way, distance_to_intersection, distance_to_end_route, total_distance_walked]
  sorry

end NUMINAMATH_GPT_rich_walked_distance_l1792_179242


namespace NUMINAMATH_GPT_find_unknown_rate_of_two_blankets_l1792_179247

-- Definitions of conditions based on the problem statement
def purchased_blankets_at_100 : Nat := 3
def price_per_blanket_at_100 : Nat := 100
def total_cost_at_100 := purchased_blankets_at_100 * price_per_blanket_at_100

def purchased_blankets_at_150 : Nat := 3
def price_per_blanket_at_150 : Nat := 150
def total_cost_at_150 := purchased_blankets_at_150 * price_per_blanket_at_150

def purchased_blankets_at_x : Nat := 2
def blankets_total : Nat := 8
def average_price : Nat := 150
def total_cost := blankets_total * average_price

-- The proof statement
theorem find_unknown_rate_of_two_blankets (x : Nat) 
  (h : purchased_blankets_at_100 * price_per_blanket_at_100 + 
       purchased_blankets_at_150 * price_per_blanket_at_150 + 
       purchased_blankets_at_x * x = total_cost) : x = 225 :=
by sorry

end NUMINAMATH_GPT_find_unknown_rate_of_two_blankets_l1792_179247


namespace NUMINAMATH_GPT_maximum_distance_area_of_ring_l1792_179206

def num_radars : ℕ := 9
def radar_radius : ℝ := 37
def ring_width : ℝ := 24

theorem maximum_distance (θ : ℝ) (hθ : θ = 20) 
  : (∀ d, d = radar_radius * (ring_width / 2 / (radar_radius^2 - (ring_width / 2)^2).sqrt)) →
    ( ∀ dist_from_center, dist_from_center = radar_radius / θ.sin) :=
sorry

theorem area_of_ring (θ : ℝ) (hθ : θ = 20) 
  : (∀ a, a = π * (ring_width * radar_radius * 2 / θ.tan)) →
    ( ∀ area, area = 1680 * π / θ.tan) :=
sorry

end NUMINAMATH_GPT_maximum_distance_area_of_ring_l1792_179206


namespace NUMINAMATH_GPT_overall_percentage_l1792_179249

theorem overall_percentage (s1 s2 s3 : ℝ) (h1 : s1 = 60) (h2 : s2 = 80) (h3 : s3 = 85) :
  (s1 + s2 + s3) / 3 = 75 := by
  sorry

end NUMINAMATH_GPT_overall_percentage_l1792_179249


namespace NUMINAMATH_GPT_simplify_expression_l1792_179270

theorem simplify_expression (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h : a^4 + b^4 = a^2 + b^2) :
  (a / b + b / a - 1 / (a * b)) = 3 :=
  sorry

end NUMINAMATH_GPT_simplify_expression_l1792_179270


namespace NUMINAMATH_GPT_line_parabola_one_intersection_not_tangent_l1792_179201

theorem line_parabola_one_intersection_not_tangent {A B C D : ℝ} (h: ∀ x : ℝ, ((A * x ^ 2 + B * x + C) = D) → False) :
  ¬ ∃ x : ℝ, (A * x ^ 2 + B * x + C) = D ∧ 2 * x * A + B = 0 := sorry

end NUMINAMATH_GPT_line_parabola_one_intersection_not_tangent_l1792_179201


namespace NUMINAMATH_GPT_no_k_satisfying_condition_l1792_179283

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_k_satisfying_condition :
  ∀ k : ℕ, (∃ p q : ℕ, p ≠ q ∧ is_prime p ∧ is_prime q ∧ k = p * q ∧ p + q = 71) → false :=
by
  sorry

end NUMINAMATH_GPT_no_k_satisfying_condition_l1792_179283


namespace NUMINAMATH_GPT_eighth_odd_multiple_of_5_is_75_l1792_179216

theorem eighth_odd_multiple_of_5_is_75 : ∃ n : ℕ, (n > 0 ∧ n % 2 = 1 ∧ n % 5 = 0 ∧ ∃ k : ℕ, k = 8 ∧ n = 10 * k - 5) :=
  sorry

end NUMINAMATH_GPT_eighth_odd_multiple_of_5_is_75_l1792_179216


namespace NUMINAMATH_GPT_mean_of_five_numbers_is_correct_l1792_179250

-- Define the sum of the five numbers
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers
def number_of_numbers : ℚ := 5

-- Define the mean
def mean_of_five_numbers := sum_of_five_numbers / number_of_numbers

-- State the theorem
theorem mean_of_five_numbers_is_correct : mean_of_five_numbers = 3 / 20 :=
by
  -- The proof is omitted, use sorry to indicate this.
  sorry

end NUMINAMATH_GPT_mean_of_five_numbers_is_correct_l1792_179250


namespace NUMINAMATH_GPT_candies_total_l1792_179290

-- Defining the given conditions
def LindaCandies : ℕ := 34
def ChloeCandies : ℕ := 28
def TotalCandies : ℕ := LindaCandies + ChloeCandies

-- Proving the total number of candies
theorem candies_total : TotalCandies = 62 :=
  by
    sorry

end NUMINAMATH_GPT_candies_total_l1792_179290


namespace NUMINAMATH_GPT_power_function_analysis_l1792_179296

theorem power_function_analysis (f : ℝ → ℝ) (α : ℝ) (h : ∀ x > 0, f x = x ^ α) (h_f : f 9 = 3) :
  (∀ x ≥ 0, f x = x ^ (1 / 2)) ∧
  (∀ x ≥ 4, f x ≥ 2) ∧
  (∀ x1 x2 : ℝ, x2 > x1 ∧ x1 > 0 → (f (x1) + f (x2)) / 2 < f ((x1 + x2) / 2)) :=
by
  -- Solution steps would go here
  sorry

end NUMINAMATH_GPT_power_function_analysis_l1792_179296


namespace NUMINAMATH_GPT_mike_went_to_last_year_l1792_179215

def this_year_games : ℕ := 15
def games_missed_this_year : ℕ := 41
def total_games_attended : ℕ := 54
def last_year_games : ℕ := total_games_attended - this_year_games

theorem mike_went_to_last_year :
  last_year_games = 39 :=
  by sorry

end NUMINAMATH_GPT_mike_went_to_last_year_l1792_179215


namespace NUMINAMATH_GPT_ratio_p_r_l1792_179241

     variables (p q r s : ℚ)

     -- Given conditions
     def ratio_p_q := p / q = 3 / 5
     def ratio_r_s := r / s = 5 / 4
     def ratio_s_q := s / q = 1 / 3

     -- Statement to be proved
     theorem ratio_p_r 
       (h1 : ratio_p_q p q)
       (h2 : ratio_r_s r s) 
       (h3 : ratio_s_q s q) : 
       p / r = 36 / 25 :=
     sorry
     
end NUMINAMATH_GPT_ratio_p_r_l1792_179241


namespace NUMINAMATH_GPT_probability_of_last_two_marbles_one_green_one_red_l1792_179245

theorem probability_of_last_two_marbles_one_green_one_red : 
    let total_marbles := 10
    let blue := 4
    let white := 3
    let red := 2
    let green := 1
    let total_ways := Nat.choose total_marbles 8
    let favorable_ways := Nat.choose (total_marbles - red - green) 6
    total_ways = 45 ∧ favorable_ways = 28 →
    (favorable_ways : ℚ) / total_ways = 28 / 45 :=
by
    intros total_marbles blue white red green total_ways favorable_ways h
    sorry

end NUMINAMATH_GPT_probability_of_last_two_marbles_one_green_one_red_l1792_179245


namespace NUMINAMATH_GPT_fourth_square_area_l1792_179237

theorem fourth_square_area (AB BC CD AD AC : ℝ) (h1 : AB^2 = 25) (h2 : BC^2 = 49) (h3 : CD^2 = 64) (h4 : AC^2 = AB^2 + BC^2)
  (h5 : AD^2 = AC^2 + CD^2) : AD^2 = 138 :=
by
  sorry

end NUMINAMATH_GPT_fourth_square_area_l1792_179237


namespace NUMINAMATH_GPT_system_of_equations_solution_l1792_179280

theorem system_of_equations_solution (x y : ℤ) 
  (h1 : x^2 + x * y + y^2 = 37) 
  (h2 : x^4 + x^2 * y^2 + y^4 = 481) : 
  (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3) ∨ (x = -3 ∧ y = -4) ∨ (x = -4 ∧ y = -3) := 
by sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1792_179280


namespace NUMINAMATH_GPT_pen_rubber_length_difference_l1792_179277

theorem pen_rubber_length_difference (P R : ℕ) 
    (h1 : P = R + 3)
    (h2 : P = 12 - 2) 
    (h3 : R + P + 12 = 29) : 
    P - R = 3 :=
  sorry

end NUMINAMATH_GPT_pen_rubber_length_difference_l1792_179277


namespace NUMINAMATH_GPT_carbon_paper_count_l1792_179233

theorem carbon_paper_count (x : ℕ) (sheets : ℕ) (copies : ℕ) (h1 : sheets = 3) (h2 : copies = 2) :
  x = 1 :=
sorry

end NUMINAMATH_GPT_carbon_paper_count_l1792_179233


namespace NUMINAMATH_GPT_cookies_per_kid_l1792_179240

theorem cookies_per_kid (total_calories_per_lunch : ℕ) (burger_calories : ℕ) (carrot_calories_per_stick : ℕ) (num_carrot_sticks : ℕ) (cookie_calories : ℕ) (num_cookies : ℕ) : 
  total_calories_per_lunch = 750 →
  burger_calories = 400 →
  carrot_calories_per_stick = 20 →
  num_carrot_sticks = 5 →
  cookie_calories = 50 →
  num_cookies = (total_calories_per_lunch - (burger_calories + num_carrot_sticks * carrot_calories_per_stick)) / cookie_calories →
  num_cookies = 5 :=
by
  sorry

end NUMINAMATH_GPT_cookies_per_kid_l1792_179240


namespace NUMINAMATH_GPT_intersection_M_N_l1792_179234

-- Define the set M and N
def M : Set ℝ := { x | x^2 ≤ 1 }
def N : Set ℝ := {-2, 0, 1}

-- Theorem stating that the intersection of M and N is {0, 1}
theorem intersection_M_N : M ∩ N = {0, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1792_179234


namespace NUMINAMATH_GPT_age_problem_l1792_179227

theorem age_problem (A B : ℕ) 
  (h1 : A + 10 = 2 * (B - 10))
  (h2 : A = B + 12) :
  B = 42 :=
sorry

end NUMINAMATH_GPT_age_problem_l1792_179227


namespace NUMINAMATH_GPT_chapters_per_day_l1792_179210

theorem chapters_per_day (total_pages : ℕ) (total_chapters : ℕ) (total_days : ℕ)
  (h1 : total_pages = 193)
  (h2 : total_chapters = 15)
  (h3 : total_days = 660) :
  (total_chapters : ℝ) / total_days = 0.0227 :=
by 
  sorry

end NUMINAMATH_GPT_chapters_per_day_l1792_179210


namespace NUMINAMATH_GPT_part1_part2_part3_l1792_179299

def set_A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def set_B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def set_C : Set ℝ := {x | x^2 + 2 * x - 8 = 0}

theorem part1 (a : ℝ) : (set_A a ∩ set_B) = (set_A a ∪ set_B) → a = 5 :=
by
  sorry

theorem part2 (a : ℝ) : (∅ ⊂ (set_A a ∩ set_B)) ∧ (set_A a ∩ set_C = ∅) → a = -2 :=
by
  sorry

theorem part3 (a : ℝ) : (set_A a ∩ set_B) = (set_A a ∩ set_C) ∧ (set_A a ∩ set_B ≠ ∅) → a = -3 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l1792_179299


namespace NUMINAMATH_GPT_inequality_three_variables_l1792_179202

theorem inequality_three_variables (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) : 
  (1/x) + (1/y) + (1/z) ≥ 9 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_three_variables_l1792_179202


namespace NUMINAMATH_GPT_ptolemys_theorem_l1792_179286

-- Definition of the variables describing the lengths of the sides and diagonals
variables {a b c d m n : ℝ}

-- We declare that they belong to a cyclic quadrilateral
def cyclic_quadrilateral (a b c d m n : ℝ) : Prop :=
∃ (A B C D : ℝ), 
  A + C = 180 ∧ 
  B + D = 180 ∧ 
  m = (A * C) ∧ 
  n = (B * D) ∧ 
  a = (A * B) ∧ 
  b = (B * C) ∧ 
  c = (C * D) ∧ 
  d = (D * A)

-- The theorem statement in Lean form
theorem ptolemys_theorem (h : cyclic_quadrilateral a b c d m n) : m * n = a * c + b * d :=
sorry

end NUMINAMATH_GPT_ptolemys_theorem_l1792_179286


namespace NUMINAMATH_GPT_wardrobe_single_discount_l1792_179269

theorem wardrobe_single_discount :
  let p : ℝ := 50
  let d1 : ℝ := 0.30
  let d2 : ℝ := 0.20
  let final_price := p * (1 - d1) * (1 - d2)
  let equivalent_discount := 1 - (final_price / p)
  equivalent_discount = 0.44 :=
by
  let p : ℝ := 50
  let d1 : ℝ := 0.30
  let d2 : ℝ := 0.20
  let final_price := p * (1 - d1) * (1 - d2)
  let equivalent_discount := 1 - (final_price / p)
  show equivalent_discount = 0.44
  sorry

end NUMINAMATH_GPT_wardrobe_single_discount_l1792_179269
