import Mathlib

namespace NUMINAMATH_GPT_operation_B_correct_l1313_131315

theorem operation_B_correct : 3 / Real.sqrt 3 = Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_operation_B_correct_l1313_131315


namespace NUMINAMATH_GPT_new_container_volume_l1313_131302

def volume_of_cube (s : ℝ) : ℝ := s^3

theorem new_container_volume (s : ℝ) (h : volume_of_cube s = 4) : 
  volume_of_cube (2 * s) * volume_of_cube (3 * s) * volume_of_cube (4 * s) = 96 :=
by
  sorry

end NUMINAMATH_GPT_new_container_volume_l1313_131302


namespace NUMINAMATH_GPT_max_rectangle_area_l1313_131359

theorem max_rectangle_area (perimeter : ℝ) (h : perimeter = 100) : 
  ∃ (a b : ℝ), 2 * a + 2 * b = perimeter ∧ a * b = 625 :=
by
  sorry

end NUMINAMATH_GPT_max_rectangle_area_l1313_131359


namespace NUMINAMATH_GPT_polynomial_remainder_x1012_l1313_131331

theorem polynomial_remainder_x1012 (x : ℂ) : 
  (x^1012) % (x^3 - x^2 + x - 1) = 1 :=
sorry

end NUMINAMATH_GPT_polynomial_remainder_x1012_l1313_131331


namespace NUMINAMATH_GPT_Tim_scored_30_l1313_131390

-- Definitions and conditions
variables (Joe Tim Ken : ℕ)
variables (h1 : Tim = Joe + 20)
variables (h2 : Tim = Nat.div (Ken * 2) 2)
variables (h3 : Joe + Tim + Ken = 100)

-- Statement to prove
theorem Tim_scored_30 : Tim = 30 :=
by sorry

end NUMINAMATH_GPT_Tim_scored_30_l1313_131390


namespace NUMINAMATH_GPT_C_share_correct_l1313_131306

noncomputable def C_share (B_invest: ℝ) (total_profit: ℝ) : ℝ :=
  let A_invest := 3 * B_invest
  let C_invest := (3 * B_invest) * (3/2)
  let total_invest := (3 * B_invest + B_invest + C_invest)
  (C_invest / total_invest) * total_profit

theorem C_share_correct (B_invest total_profit: ℝ) 
  (hA : ∀ x: ℝ, A_invest = 3 * x)
  (hC : ∀ x: ℝ, C_invest = (3 * x) * (3/2)) :
  C_share B_invest 12375 = 6551.47 :=
by
  sorry

end NUMINAMATH_GPT_C_share_correct_l1313_131306


namespace NUMINAMATH_GPT_production_days_l1313_131330

theorem production_days (n : ℕ) (h1 : (40 * n + 90) / (n + 1) = 45) : n = 9 :=
by
  sorry

end NUMINAMATH_GPT_production_days_l1313_131330


namespace NUMINAMATH_GPT_find_supplementary_angle_l1313_131307

noncomputable def degree (x : ℝ) : ℝ := x
noncomputable def complementary_angle (x : ℝ) : ℝ := 90 - x
noncomputable def supplementary_angle (x : ℝ) : ℝ := 180 - x

theorem find_supplementary_angle
  (x : ℝ)
  (h1 : degree x / complementary_angle x = 1 / 8) :
  supplementary_angle x = 170 :=
by
  sorry

end NUMINAMATH_GPT_find_supplementary_angle_l1313_131307


namespace NUMINAMATH_GPT_duration_of_time_l1313_131308

variable (A B C : String)
variable {a1 : A = "Get up at 6:30"}
variable {b1 : B = "School ends at 3:40"}
variable {c1 : C = "It took 30 minutes to do the homework"}

theorem duration_of_time : C = "It took 30 minutes to do the homework" :=
  sorry

end NUMINAMATH_GPT_duration_of_time_l1313_131308


namespace NUMINAMATH_GPT_line_A1_A2_condition_plane_A1_A2_A3_condition_plane_through_A3_A4_parallel_to_A1_A2_condition_l1313_131377

section BarycentricCoordinates

variables {A1 A2 A3 A4 : Type} 

def barycentric_condition (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 + x2 + x3 + x4 = 1

theorem line_A1_A2_condition (x1 x2 x3 x4 : ℝ) : 
  barycentric_condition x1 x2 x3 x4 → (x3 = 0 ∧ x4 = 0) ↔ (x1 + x2 = 1) :=
by
  sorry

theorem plane_A1_A2_A3_condition (x1 x2 x3 x4 : ℝ) :
  barycentric_condition x1 x2 x3 x4 → (x4 = 0) ↔ (x1 + x2 + x3 = 1) :=
by
  sorry

theorem plane_through_A3_A4_parallel_to_A1_A2_condition (x1 x2 x3 x4 : ℝ) :
  barycentric_condition x1 x2 x3 x4 → (x1 = -x2 ∧ x3 + x4 = 1) ↔ (x1 + x2 + x3 + x4 = 1) :=
by
  sorry

end BarycentricCoordinates

end NUMINAMATH_GPT_line_A1_A2_condition_plane_A1_A2_A3_condition_plane_through_A3_A4_parallel_to_A1_A2_condition_l1313_131377


namespace NUMINAMATH_GPT_number_of_rocks_in_bucket_l1313_131318

noncomputable def average_weight_rock : ℝ := 1.5
noncomputable def total_money_made : ℝ := 60
noncomputable def price_per_pound : ℝ := 4

theorem number_of_rocks_in_bucket : 
  let total_weight_rocks := total_money_made / price_per_pound
  let number_of_rocks := total_weight_rocks / average_weight_rock
  number_of_rocks = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_rocks_in_bucket_l1313_131318


namespace NUMINAMATH_GPT_parallel_vectors_k_l1313_131311

theorem parallel_vectors_k (k : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2 - k, 3)) (h₂ : b = (2, -6)) (h₃ : a.1 * b.2 = a.2 * b.1) : k = 3 :=
sorry

end NUMINAMATH_GPT_parallel_vectors_k_l1313_131311


namespace NUMINAMATH_GPT_lemma2_l1313_131350

noncomputable def f (x a b : ℝ) := |x + a| - |x - b|

lemma lemma1 {x : ℝ} : f x 1 2 > 2 ↔ x > 3 / 2 := 
sorry

theorem lemma2 {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : ∀ x : ℝ, f x a b ≤ 3):
  1 / a + 2 / b = (1 / 3) * (3 + 2 * Real.sqrt 2) := 
sorry

end NUMINAMATH_GPT_lemma2_l1313_131350


namespace NUMINAMATH_GPT_sum_of_a_b_c_d_l1313_131338

theorem sum_of_a_b_c_d (a b c d : ℝ) (h1 : c + d = 12 * a) (h2 : c * d = -13 * b) (h3 : a + b = 12 * c) (h4 : a * b = -13 * d) (h_distinct : a ≠ c) : a + b + c + d = 2028 :=
  by 
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_sum_of_a_b_c_d_l1313_131338


namespace NUMINAMATH_GPT_solution1_solution2_l1313_131335

namespace MathProofProblem

-- Define the first system of equations
def system1 (x y : ℝ) : Prop :=
  4 * x - 2 * y = 14 ∧ 3 * x + 2 * y = 7

-- Prove the solution for the first system
theorem solution1 : ∃ (x y : ℝ), system1 x y ∧ x = 3 ∧ y = -1 := by
  sorry

-- Define the second system of equations
def system2 (x y : ℝ) : Prop :=
  y = x + 1 ∧ 2 * x + y = 10

-- Prove the solution for the second system
theorem solution2 : ∃ (x y : ℝ), system2 x y ∧ x = 3 ∧ y = 4 := by
  sorry

end MathProofProblem

end NUMINAMATH_GPT_solution1_solution2_l1313_131335


namespace NUMINAMATH_GPT_vasya_has_more_fanta_l1313_131351

-- Definitions based on the conditions:
def initial_fanta_vasya (a : ℝ) : ℝ := a
def initial_fanta_petya (a : ℝ) : ℝ := 1.1 * a
def remaining_fanta_vasya (a : ℝ) : ℝ := a * 0.98
def remaining_fanta_petya (a : ℝ) : ℝ := 1.1 * a * 0.89

-- The theorem to prove Vasya has more Fanta left than Petya.
theorem vasya_has_more_fanta (a : ℝ) (h : 0 < a) : remaining_fanta_vasya a > remaining_fanta_petya a := by
  sorry

end NUMINAMATH_GPT_vasya_has_more_fanta_l1313_131351


namespace NUMINAMATH_GPT_sum_of_ages_l1313_131393

theorem sum_of_ages (S F : ℕ) 
  (h1 : F - 18 = 3 * (S - 18)) 
  (h2 : F = 2 * S) : S + F = 108 := by 
  sorry

end NUMINAMATH_GPT_sum_of_ages_l1313_131393


namespace NUMINAMATH_GPT_inequality_proof_l1313_131368

variable (a b : ℝ)

theorem inequality_proof (h1 : -1 < b) (h2 : b < 0) (h3 : a < 0) : 
  (a * b > a * b^2) ∧ (a * b^2 > a) := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1313_131368


namespace NUMINAMATH_GPT_num_male_rabbits_l1313_131313

/-- 
There are 12 white rabbits and 9 black rabbits. 
There are 8 female rabbits. 
Prove that the number of male rabbits is 13.
-/
theorem num_male_rabbits (white_rabbits : ℕ) (black_rabbits : ℕ) (female_rabbits: ℕ) 
  (h_white : white_rabbits = 12) (h_black : black_rabbits = 9) (h_female : female_rabbits = 8) :
  (white_rabbits + black_rabbits - female_rabbits = 13) :=
by
  sorry

end NUMINAMATH_GPT_num_male_rabbits_l1313_131313


namespace NUMINAMATH_GPT_fraction_of_quarters_from_1800_to_1809_l1313_131360

def num_total_quarters := 26
def num_states_1800s := 8

theorem fraction_of_quarters_from_1800_to_1809 : 
  (num_states_1800s / num_total_quarters : ℚ) = 4 / 13 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_quarters_from_1800_to_1809_l1313_131360


namespace NUMINAMATH_GPT_cube_edge_ratio_l1313_131365

theorem cube_edge_ratio (a b : ℕ) (h : a^3 = 27 * b^3) : a = 3 * b :=
sorry

end NUMINAMATH_GPT_cube_edge_ratio_l1313_131365


namespace NUMINAMATH_GPT_ny_mets_fans_l1313_131348

-- Let Y be the number of NY Yankees fans
-- Let M be the number of NY Mets fans
-- Let R be the number of Boston Red Sox fans
variables (Y M R : ℕ)

-- Given conditions
def ratio_Y_M : Prop := 3 * M = 2 * Y
def ratio_M_R : Prop := 4 * R = 5 * M
def total_fans : Prop := Y + M + R = 330

-- The theorem to prove
theorem ny_mets_fans (h1 : ratio_Y_M Y M) (h2 : ratio_M_R M R) (h3 : total_fans Y M R) : M = 88 :=
sorry

end NUMINAMATH_GPT_ny_mets_fans_l1313_131348


namespace NUMINAMATH_GPT_emily_gave_away_l1313_131366

variable (x : ℕ)

def emily_initial_books : ℕ := 7

def emily_books_after_giving_away (x : ℕ) : ℕ := 7 - x

def emily_books_after_buying_more (x : ℕ) : ℕ :=
  7 - x + 14

def emily_final_books : ℕ := 19

theorem emily_gave_away : (emily_books_after_buying_more x = emily_final_books) → x = 2 := by
  sorry

end NUMINAMATH_GPT_emily_gave_away_l1313_131366


namespace NUMINAMATH_GPT_curve_equation_l1313_131370

theorem curve_equation :
  (∃ (x y : ℝ), 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0 ∧ x = 3 ∧ y = 2) ∧
  (∃ (C : ℝ), 
    8 * 3 + 6 * 2 + C = 0 ∧
    8 * x + 6 * y + C = 0 ∧
    4 * x + 3 * y - 18 = 0 ∧
    ∀ x y, 6 * x - 8 * y + 3 = 0 → 
    4 * x + 3 * y - 18 = 0) ∧
  (∃ (a : ℝ), ∀ x y, (x + 1)^2 + 1 = (x - 1)^2 + 9 →
    ((x - 2)^2 + y^2 = 10 ∧ a = 2)) :=
sorry

end NUMINAMATH_GPT_curve_equation_l1313_131370


namespace NUMINAMATH_GPT_find_a_l1313_131326

noncomputable def f (x : ℝ) := x^2

theorem find_a (a : ℝ) (h : (1/2) * a^2 * (a/2) = 2) :
  a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1313_131326


namespace NUMINAMATH_GPT_probability_of_yellow_jelly_bean_l1313_131375

theorem probability_of_yellow_jelly_bean (P_red P_orange P_yellow : ℝ) 
  (h1 : P_red = 0.2) 
  (h2 : P_orange = 0.5) 
  (h3 : P_red + P_orange + P_yellow = 1) : 
  P_yellow = 0.3 :=
sorry

end NUMINAMATH_GPT_probability_of_yellow_jelly_bean_l1313_131375


namespace NUMINAMATH_GPT_smallest_x_multiple_of_53_l1313_131346

theorem smallest_x_multiple_of_53 : ∃ (x : Nat), (x > 0) ∧ ( ∀ (n : Nat), (n > 0) ∧ ((3 * n + 43) % 53 = 0) → x ≤ n ) ∧ ((3 * x + 43) % 53 = 0) :=
sorry

end NUMINAMATH_GPT_smallest_x_multiple_of_53_l1313_131346


namespace NUMINAMATH_GPT_find_x_l1313_131327

theorem find_x :
  ∃ x : ℕ, x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7 ∧ (∀ y : ℕ, y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7 → y ≥ x) :=
sorry

end NUMINAMATH_GPT_find_x_l1313_131327


namespace NUMINAMATH_GPT_r4_plus_inv_r4_l1313_131324

theorem r4_plus_inv_r4 (r : ℝ) (h : (r + (1 : ℝ) / r) ^ 2 = 5) : r ^ 4 + (1 : ℝ) / r ^ 4 = 7 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_r4_plus_inv_r4_l1313_131324


namespace NUMINAMATH_GPT_airplane_time_in_air_l1313_131392

-- Define conditions
def distance_seaport_island := 840  -- Total distance in km
def speed_icebreaker := 20          -- Speed of the icebreaker in km/h
def time_icebreaker := 22           -- Total time the icebreaker traveled in hours
def speed_airplane := 120           -- Speed of the airplane in km/h

-- Prove the time the airplane spent in the air
theorem airplane_time_in_air : (distance_seaport_island - speed_icebreaker * time_icebreaker) / speed_airplane = 10 / 3 := by
  -- This is where the proof steps would go, but we're placing sorry to skip it for now.
  sorry

end NUMINAMATH_GPT_airplane_time_in_air_l1313_131392


namespace NUMINAMATH_GPT_average_speed_l1313_131300

theorem average_speed (D T : ℝ) (h1 : D = 100) (h2 : T = 6) : (D / T) = 50 / 3 := by
  sorry

end NUMINAMATH_GPT_average_speed_l1313_131300


namespace NUMINAMATH_GPT_flynn_tv_weeks_l1313_131320

-- Define the conditions
def minutes_per_weekday := 30
def additional_hours_weekend := 2
def total_hours := 234
def minutes_per_hour := 60
def weekdays := 5

-- Define the total watching time per week in minutes
def total_weekday_minutes := minutes_per_weekday * weekdays
def total_weekday_hours := total_weekday_minutes / minutes_per_hour
def total_weekly_hours := total_weekday_hours + additional_hours_weekend

-- Create a theorem to prove the correct number of weeks
theorem flynn_tv_weeks : 
  (total_hours / total_weekly_hours) = 52 := 
by
  sorry

end NUMINAMATH_GPT_flynn_tv_weeks_l1313_131320


namespace NUMINAMATH_GPT_largest_natural_gas_reserves_l1313_131355
noncomputable def top_country_in_natural_gas_reserves : String :=
  "Russia"

theorem largest_natural_gas_reserves (countries : Fin 4 → String) :
  countries 0 = "Russia" → 
  countries 1 = "Finland" → 
  countries 2 = "United Kingdom" → 
  countries 3 = "Norway" → 
  top_country_in_natural_gas_reserves = countries 0 :=
by
  intros h_russia h_finland h_uk h_norway
  rw [h_russia]
  sorry

end NUMINAMATH_GPT_largest_natural_gas_reserves_l1313_131355


namespace NUMINAMATH_GPT_find_subtracted_value_l1313_131340

-- Define the conditions
def chosen_number := 124
def result := 110

-- Lean statement to prove
theorem find_subtracted_value (x : ℕ) (y : ℕ) (h1 : x = chosen_number) (h2 : 2 * x - y = result) : y = 138 :=
by
  sorry

end NUMINAMATH_GPT_find_subtracted_value_l1313_131340


namespace NUMINAMATH_GPT_otgaday_wins_l1313_131357

theorem otgaday_wins (a n : ℝ) : a * n > 0.91 * a * n := 
by
  sorry

end NUMINAMATH_GPT_otgaday_wins_l1313_131357


namespace NUMINAMATH_GPT_sin_120_eq_sqrt3_div_2_l1313_131378

theorem sin_120_eq_sqrt3_div_2
  (h1 : 120 = 180 - 60)
  (h2 : ∀ θ, Real.sin (180 - θ) = Real.sin θ)
  (h3 : Real.sin 60 = Real.sqrt 3 / 2) :
  Real.sin 120 = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_sin_120_eq_sqrt3_div_2_l1313_131378


namespace NUMINAMATH_GPT_find_a_l1313_131376

theorem find_a (a : ℝ) :
  let θ := 120
  let tan120 := -Real.sqrt 3
  (∀ x y: ℝ, 2 * x + a * y + 3 = 0) →
  a = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1313_131376


namespace NUMINAMATH_GPT_find_m_real_find_m_imaginary_l1313_131364

-- Define the real part condition
def real_part_condition (m : ℝ) : Prop :=
  m^2 - 3 * m - 4 = 0

-- Define the imaginary part condition
def imaginary_part_condition (m : ℝ) : Prop :=
  m^2 - 2 * m - 3 = 0 ∧ m^2 - 3 * m - 4 ≠ 0

-- Theorem for the first part
theorem find_m_real : ∀ (m : ℝ), (real_part_condition m) → (m = 4 ∨ m = -1) :=
by sorry

-- Theorem for the second part
theorem find_m_imaginary : ∀ (m : ℝ), (imaginary_part_condition m) → (m = 3) :=
by sorry

end NUMINAMATH_GPT_find_m_real_find_m_imaginary_l1313_131364


namespace NUMINAMATH_GPT_power_expression_l1313_131303

theorem power_expression : (1 / ((-5)^4)^2) * (-5)^9 = -5 := sorry

end NUMINAMATH_GPT_power_expression_l1313_131303


namespace NUMINAMATH_GPT_different_people_count_l1313_131389

def initial_people := 9
def people_left := 6
def people_joined := 3
def total_different_people (initial_people people_left people_joined : ℕ) : ℕ :=
  initial_people + people_joined

theorem different_people_count :
  total_different_people initial_people people_left people_joined = 12 :=
by
  sorry

end NUMINAMATH_GPT_different_people_count_l1313_131389


namespace NUMINAMATH_GPT_taxi_fare_l1313_131358

theorem taxi_fare (x : ℝ) : 
  (2.40 + 2 * (x - 0.5) = 8) → x = 3.3 := by
  sorry

end NUMINAMATH_GPT_taxi_fare_l1313_131358


namespace NUMINAMATH_GPT_find_m_n_sum_l1313_131316

theorem find_m_n_sum (m n : ℝ) :
  ( ∀ x, -3 < x ∧ x < 6 → x^2 - m * x - 6 * n < 0 ) →
  m + n = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_m_n_sum_l1313_131316


namespace NUMINAMATH_GPT_factor_difference_of_squares_l1313_131395

theorem factor_difference_of_squares (x : ℝ) : 49 - 16 * x^2 = (7 - 4 * x) * (7 + 4 * x) :=
by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l1313_131395


namespace NUMINAMATH_GPT_valid_sandwiches_bob_can_order_l1313_131321

def total_breads := 5
def total_meats := 7
def total_cheeses := 6

def undesired_combinations_count : Nat :=
  let turkey_swiss := total_breads
  let roastbeef_rye := total_cheeses
  let roastbeef_swiss := total_breads
  turkey_swiss + roastbeef_rye + roastbeef_swiss

def total_sandwiches : Nat :=
  total_breads * total_meats * total_cheeses

def valid_sandwiches_count : Nat :=
  total_sandwiches - undesired_combinations_count

theorem valid_sandwiches_bob_can_order : valid_sandwiches_count = 194 := by
  sorry

end NUMINAMATH_GPT_valid_sandwiches_bob_can_order_l1313_131321


namespace NUMINAMATH_GPT_largest_nonrepresentable_integer_l1313_131385

theorem largest_nonrepresentable_integer :
  (∀ a b : ℕ, 8 * a + 15 * b ≠ 97) ∧ (∀ n : ℕ, n > 97 → ∃ a b : ℕ, n = 8 * a + 15 * b) :=
sorry

end NUMINAMATH_GPT_largest_nonrepresentable_integer_l1313_131385


namespace NUMINAMATH_GPT_ceil_sqrt_of_900_l1313_131317

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem ceil_sqrt_of_900 :
  isPerfectSquare 36 ∧ isPerfectSquare 25 ∧ (36 * 25 = 900) → 
  Int.ceil (Real.sqrt 900) = 30 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ceil_sqrt_of_900_l1313_131317


namespace NUMINAMATH_GPT_largest_inscribed_rightangled_parallelogram_l1313_131374

theorem largest_inscribed_rightangled_parallelogram (r : ℝ) (x y : ℝ) 
  (parallelogram_inscribed : x = 2 * r * Real.sin (45 * π / 180) ∧ y = 2 * r * Real.cos (45 * π / 180)) :
  x = r * Real.sqrt 2 ∧ y = r * Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_largest_inscribed_rightangled_parallelogram_l1313_131374


namespace NUMINAMATH_GPT_birthday_friends_count_l1313_131336

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end NUMINAMATH_GPT_birthday_friends_count_l1313_131336


namespace NUMINAMATH_GPT_value_of_k_l1313_131310

theorem value_of_k (x z k : ℝ) (h1 : 2 * x - (-1) + 3 * z = 9) 
                   (h2 : x + 2 * (-1) - z = k) 
                   (h3 : -x + (-1) + 4 * z = 6) : 
                   k = -3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_l1313_131310


namespace NUMINAMATH_GPT_heights_on_equal_sides_are_equal_l1313_131352

-- Given conditions as definitions
def is_isosceles_triangle (a b c : ℝ) := (a = b ∨ b = c ∨ c = a)
def height_on_equal_sides_equal (a b c : ℝ) := is_isosceles_triangle a b c → a = b

-- Lean theorem statement to prove
theorem heights_on_equal_sides_are_equal {a b c : ℝ} : is_isosceles_triangle a b c → height_on_equal_sides_equal a b c := 
sorry

end NUMINAMATH_GPT_heights_on_equal_sides_are_equal_l1313_131352


namespace NUMINAMATH_GPT_James_has_43_Oreos_l1313_131304

variable (J : ℕ)
variable (James_Oreos : ℕ)

-- Conditions
def condition1 : Prop := James_Oreos = 4 * J + 7
def condition2 : Prop := J + James_Oreos = 52

-- The statement to prove: James has 43 Oreos given the conditions
theorem James_has_43_Oreos (h1 : condition1 J James_Oreos) (h2 : condition2 J James_Oreos) : James_Oreos = 43 :=
by
  sorry

end NUMINAMATH_GPT_James_has_43_Oreos_l1313_131304


namespace NUMINAMATH_GPT_population_increase_difference_l1313_131333

noncomputable def births_per_day : ℝ := 24 / 6
noncomputable def deaths_per_day : ℝ := 24 / 16
noncomputable def net_increase_per_day : ℝ := births_per_day - deaths_per_day
noncomputable def annual_increase_regular_year : ℝ := net_increase_per_day * 365
noncomputable def annual_increase_leap_year : ℝ := net_increase_per_day * 366

theorem population_increase_difference :
  annual_increase_leap_year - annual_increase_regular_year = 2.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_population_increase_difference_l1313_131333


namespace NUMINAMATH_GPT_tan_product_min_value_l1313_131384

theorem tan_product_min_value (α β γ : ℝ) (h1 : α > 0 ∧ α < π / 2) 
    (h2 : β > 0 ∧ β < π / 2) (h3 : γ > 0 ∧ γ < π / 2)
    (h4 : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) : 
  (Real.tan α * Real.tan β * Real.tan γ) = 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_tan_product_min_value_l1313_131384


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l1313_131322

theorem algebraic_expression_evaluation (x y : ℤ) (h1 : x = -2) (h2 : y = -4) : 2 * x^2 - y + 3 = 15 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l1313_131322


namespace NUMINAMATH_GPT_angle_is_60_degrees_l1313_131309

-- Definitions
def angle_is_twice_complementary (x : ℝ) : Prop := x = 2 * (90 - x)

-- Theorem statement
theorem angle_is_60_degrees (x : ℝ) (h : angle_is_twice_complementary x) : x = 60 :=
by sorry

end NUMINAMATH_GPT_angle_is_60_degrees_l1313_131309


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1313_131361

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ) (d : ℤ),
    a 1 = 1 →
    d ≠ 0 →
    (a 2 = a 1 + d) →
    (a 3 = a 1 + 2 * d) →
    (a 6 = a 1 + 5 * d) →
    (a 3)^2 = (a 2) * (a 6) →
    (1 + 2 * d)^2 = (1 + d) * (1 + 5 * d) →
    (6 / 2) * (2 * a 1 + (6 - 1) * d) = -24 := 
by intros a d h1 h2 h3 h4 h5 h6 h7
   sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1313_131361


namespace NUMINAMATH_GPT_cameron_total_questions_l1313_131388

def usual_questions : Nat := 2

def group_a_questions : Nat := 
  let q1 := 2 * 1 -- 2 people who asked a single question each
  let q2 := 3 * usual_questions -- 3 people who asked two questions as usual
  let q3 := 1 * 5 -- 1 person who asked 5 questions
  q1 + q2 + q3

def group_b_questions : Nat :=
  let q1 := 1 * 0 -- 1 person asked no questions
  let q2 := 6 * 3 -- 6 people asked 3 questions each
  let q3 := 4 * usual_questions -- 4 people asked the usual number of questions
  q1 + q2 + q3

def group_c_questions : Nat :=
  let q1 := 1 * (usual_questions * 3) -- 1 person asked three times as many questions as usual
  let q2 := 1 * 1 -- 1 person asked only one question
  let q3 := 2 * 0 -- 2 members asked no questions
  let q4 := 4 * usual_questions -- The remaining tourists asked the usual 2 questions each
  q1 + q2 + q3 + q4

def group_d_questions : Nat :=
  let q1 := 1 * (usual_questions * 4) -- 1 individual asked four times as many questions as normal
  let q2 := 1 * 0 -- 1 person asked no questions at all
  let q3 := 3 * usual_questions -- The remaining tourists asked the usual number of questions
  q1 + q2 + q3

def group_e_questions : Nat :=
  let q1 := 3 * (usual_questions * 2) -- 3 people asked double the average number of questions
  let q2 := 2 * 0 -- 2 people asked none
  let q3 := 1 * 5 -- 1 tourist asked 5 questions
  let q4 := 3 * usual_questions -- The remaining tourists asked the usual number
  q1 + q2 + q3 + q4

def group_f_questions : Nat :=
  let q1 := 2 * 3 -- 2 individuals asked three questions each
  let q2 := 1 * 0 -- 1 person asked no questions
  let q3 := 4 * usual_questions -- The remaining tourists asked the usual number
  q1 + q2 + q3

def total_questions : Nat :=
  group_a_questions + group_b_questions + group_c_questions + group_d_questions + group_e_questions + group_f_questions

theorem cameron_total_questions : total_questions = 105 := by
  sorry

end NUMINAMATH_GPT_cameron_total_questions_l1313_131388


namespace NUMINAMATH_GPT_circle_center_eq_circle_center_is_1_3_2_l1313_131397

-- Define the problem: Given the equation of the circle, prove the center is (1, 3/2)
theorem circle_center_eq (x y : ℝ) :
  16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 100 = 0 ↔ (x - 1)^2 + (y - 3/2)^2 = 3 := sorry

-- Prove that the center of the circle from the given equation is (1, 3/2)
theorem circle_center_is_1_3_2 :
  ∃ x y : ℝ, (16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 100 = 0) ∧ (x = 1) ∧ (y = 3 / 2) := sorry

end NUMINAMATH_GPT_circle_center_eq_circle_center_is_1_3_2_l1313_131397


namespace NUMINAMATH_GPT_computer_operations_in_three_hours_l1313_131334

theorem computer_operations_in_three_hours :
  let additions_per_second := 12000
  let multiplications_per_second := 2 * additions_per_second
  let seconds_in_three_hours := 3 * 3600
  (additions_per_second + multiplications_per_second) * seconds_in_three_hours = 388800000 :=
by
  sorry

end NUMINAMATH_GPT_computer_operations_in_three_hours_l1313_131334


namespace NUMINAMATH_GPT_positive_number_and_cube_l1313_131367

theorem positive_number_and_cube (n : ℕ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 ∧ n^3 = 2744 :=
by sorry

end NUMINAMATH_GPT_positive_number_and_cube_l1313_131367


namespace NUMINAMATH_GPT_solve_dog_walking_minutes_l1313_131347

-- Definitions based on the problem conditions
def cost_one_dog (x : ℕ) : ℕ := 20 + x
def cost_two_dogs : ℕ := 54
def cost_three_dogs : ℕ := 87
def total_earnings (x : ℕ) : ℕ := cost_one_dog x + cost_two_dogs + cost_three_dogs

-- Proving that the total earnings equal to 171 implies x = 10
theorem solve_dog_walking_minutes (x : ℕ) (h : total_earnings x = 171) : x = 10 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_solve_dog_walking_minutes_l1313_131347


namespace NUMINAMATH_GPT_iterate_fixed_point_l1313_131342

theorem iterate_fixed_point {f : ℤ → ℤ} (a : ℤ) :
  (∀ n, f^[n] a = a → f a = a) ∧ (f a = a → f^[22000] a = a) :=
sorry

end NUMINAMATH_GPT_iterate_fixed_point_l1313_131342


namespace NUMINAMATH_GPT_original_triangle_area_l1313_131382

theorem original_triangle_area :
  let S_perspective := (1 / 2) * 1 * 1 * Real.sin (Real.pi / 3)
  let S_ratio := Real.sqrt 2 / 4
  let S_perspective_value := Real.sqrt 3 / 4
  let S_original := S_perspective_value / S_ratio
  S_original = Real.sqrt 6 / 2 :=
by
  sorry

end NUMINAMATH_GPT_original_triangle_area_l1313_131382


namespace NUMINAMATH_GPT_sqrt_equiv_c_d_l1313_131369

noncomputable def c : ℤ := 3
noncomputable def d : ℤ := 375

theorem sqrt_equiv_c_d : ∀ (x y : ℤ), x = 3^5 ∧ y = 5^3 → (∃ c d : ℤ, (c = 3 ∧ d = 375 ∧ x * y = c^4 * d))
    ∧ c + d = 378 := by sorry

end NUMINAMATH_GPT_sqrt_equiv_c_d_l1313_131369


namespace NUMINAMATH_GPT_x_plus_y_plus_z_equals_4_l1313_131345

theorem x_plus_y_plus_z_equals_4 (x y z : ℝ) 
  (h1 : 2 * x + 3 * y + 4 * z = 10) 
  (h2 : y + 2 * z = 2) : 
  x + y + z = 4 :=
by
  sorry

end NUMINAMATH_GPT_x_plus_y_plus_z_equals_4_l1313_131345


namespace NUMINAMATH_GPT_average_age_of_women_l1313_131372

-- Defining the conditions
def average_age_of_men : ℝ := 40
def number_of_men : ℕ := 15
def increase_in_average : ℝ := 2.9
def ages_of_replaced_men : List ℝ := [26, 32, 41, 39]
def number_of_women : ℕ := 4

-- Stating the proof problem
theorem average_age_of_women :
  let total_age_of_men := average_age_of_men * number_of_men
  let total_age_of_replaced_men := ages_of_replaced_men.sum
  let new_average_age := average_age_of_men + increase_in_average
  let new_total_age_of_group := new_average_age * number_of_men
  let total_age_of_women := new_total_age_of_group - (total_age_of_men - total_age_of_replaced_men)
  let average_age_of_women := total_age_of_women / number_of_women
  average_age_of_women = 45.375 :=
sorry

end NUMINAMATH_GPT_average_age_of_women_l1313_131372


namespace NUMINAMATH_GPT_number_of_solutions_is_3_l1313_131301

noncomputable def count_solutions : Nat :=
  Nat.card {x : Nat // x < 150 ∧ (x + 15) % 45 = 75 % 45}

theorem number_of_solutions_is_3 : count_solutions = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_solutions_is_3_l1313_131301


namespace NUMINAMATH_GPT_find_abc_square_sum_l1313_131349

theorem find_abc_square_sum (a b c : ℝ) 
  (h1 : a^2 + 3 * b = 9) 
  (h2 : b^2 + 5 * c = -8) 
  (h3 : c^2 + 7 * a = -18) : 
  a^2 + b^2 + c^2 = 20.75 := 
sorry

end NUMINAMATH_GPT_find_abc_square_sum_l1313_131349


namespace NUMINAMATH_GPT_limit_at_minus_one_third_l1313_131328

theorem limit_at_minus_one_third : 
  ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ 
  (∀ (x : ℝ), 0 < |x + 1 / 3| ∧ |x + 1 / 3| < δ → 
  |(9 * x^2 - 1) / (x + 1 / 3) + 6| < ε) :=
sorry

end NUMINAMATH_GPT_limit_at_minus_one_third_l1313_131328


namespace NUMINAMATH_GPT_proof_quadratic_conclusions_l1313_131323

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given points on the graph
def points_on_graph (a b c : ℝ) : Prop :=
  quadratic_function a b c (-1) = -2 ∧
  quadratic_function a b c 0 = -3 ∧
  quadratic_function a b c 1 = -4 ∧
  quadratic_function a b c 2 = -3 ∧
  quadratic_function a b c 3 = 0

-- Assertions based on the problem statement
def assertion_A (a b : ℝ) : Prop := 2 * a + b = 0

def assertion_C (a b c : ℝ) : Prop :=
  quadratic_function a b c 3 = 0 ∧ quadratic_function a b c (-1) = 0

def assertion_D (a b c : ℝ) (m : ℝ) (y1 y2 : ℝ) : Prop :=
  (quadratic_function a b c (m - 1) = y1) → 
  (quadratic_function a b c m = y2) → 
  (y1 < y2) → 
  (m > 3 / 2)

-- Final theorem statement to be proven
theorem proof_quadratic_conclusions (a b c : ℝ) (m y1 y2 : ℝ) :
  points_on_graph a b c →
  assertion_A a b →
  assertion_C a b c →
  assertion_D a b c m y1 y2 :=
by
  sorry

end NUMINAMATH_GPT_proof_quadratic_conclusions_l1313_131323


namespace NUMINAMATH_GPT_divisible_by_5_l1313_131339

-- Problem statement: For which values of \( x \) is \( 2^x - 1 \) divisible by \( 5 \)?
-- Equivalent Proof Problem in Lean 4.

theorem divisible_by_5 (x : ℕ) : 
  (∃ t : ℕ, x = 6 * t + 1) ∨ (∃ t : ℕ, x = 6 * t + 4) ↔ (5 ∣ (2^x - 1)) :=
by sorry

end NUMINAMATH_GPT_divisible_by_5_l1313_131339


namespace NUMINAMATH_GPT_complex_coordinates_l1313_131399

-- Define the imaginary unit
def i : ℂ := ⟨0, 1⟩

-- Define the complex number (1 + i)
def z1 : ℂ := 1 + i

-- Define the complex number i
def z2 : ℂ := i

-- The problem statement to be proven: the given complex number equals 1 - i
theorem complex_coordinates : (z1 / z2) = 1 - i :=
  sorry

end NUMINAMATH_GPT_complex_coordinates_l1313_131399


namespace NUMINAMATH_GPT_triangle_area_is_31_5_l1313_131343

def point := (ℝ × ℝ)

def A : point := (2, 3)
def B : point := (9, 3)
def C : point := (5, 12)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_is_31_5 :
  triangle_area A B C = 31.5 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_triangle_area_is_31_5_l1313_131343


namespace NUMINAMATH_GPT_alice_next_birthday_age_l1313_131371

theorem alice_next_birthday_age (a b c : ℝ) 
  (h1 : a = 1.25 * b)
  (h2 : b = 0.7 * c)
  (h3 : a + b + c = 30) : a + 1 = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_alice_next_birthday_age_l1313_131371


namespace NUMINAMATH_GPT_correct_option_C_correct_option_D_l1313_131398

-- definitions representing the conditions
def A_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≤ 0
def B_inequality (x : ℝ) : Prop := (2 * x + 1) * (3 - x) ≥ 0
def C_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def D_inequality (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0
def solution_set (x : ℝ) : Prop := (-1 / 2 ≤ x ∧ x < 3)

-- proving that option C is equivalent to the solution set
theorem correct_option_C : ∀ x : ℝ, C_inequality x ↔ solution_set x :=
by sorry

-- proving that option D is equivalent to the solution set
theorem correct_option_D : ∀ x : ℝ, D_inequality x ↔ solution_set x :=
by sorry

end NUMINAMATH_GPT_correct_option_C_correct_option_D_l1313_131398


namespace NUMINAMATH_GPT_value_of_f_a1_a3_a5_l1313_131312

-- Definitions
def monotonically_increasing (f : ℝ → ℝ) :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

def odd_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = -f x

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

-- Problem statement
theorem value_of_f_a1_a3_a5 (f : ℝ → ℝ) (a : ℕ → ℝ) :
  monotonically_increasing f →
  odd_function f →
  arithmetic_sequence a →
  a 3 > 0 →
  f (a 1) + f (a 3) + f (a 5) > 0 :=
by
  intros h_mono h_odd h_arith h_a3
  sorry

end NUMINAMATH_GPT_value_of_f_a1_a3_a5_l1313_131312


namespace NUMINAMATH_GPT_molecular_weight_H2O_correct_l1313_131356

-- Define atomic weights as constants
def atomic_weight_hydrogen : ℝ := 1.008
def atomic_weight_oxygen : ℝ := 15.999

-- Define the number of atoms in H2O
def num_hydrogens : ℕ := 2
def num_oxygens : ℕ := 1

-- Define molecular weight calculation for H2O
def molecular_weight_H2O : ℝ :=
  num_hydrogens * atomic_weight_hydrogen + num_oxygens * atomic_weight_oxygen

-- State the theorem that this molecular weight is 18.015 amu
theorem molecular_weight_H2O_correct :
  molecular_weight_H2O = 18.015 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_H2O_correct_l1313_131356


namespace NUMINAMATH_GPT_compute_f_l1313_131341

theorem compute_f (f : ℕ → ℚ) (h1 : f 1 = 1 / 3)
  (h2 : ∀ n : ℕ, n ≥ 2 → f n = (2 * (n - 1) - 1) / (2 * (n - 1) + 3) * f (n - 1)) :
  ∀ n : ℕ, n ≥ 1 → f n = 1 / ((2 * n - 1) * (2 * n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_compute_f_l1313_131341


namespace NUMINAMATH_GPT_price_of_shirt_l1313_131353

theorem price_of_shirt (T S : ℝ) 
  (h1 : T + S = 80.34) 
  (h2 : T = S - 7.43) : 
  T = 36.455 :=
by
  sorry

end NUMINAMATH_GPT_price_of_shirt_l1313_131353


namespace NUMINAMATH_GPT_solve_inequality_l1313_131329

theorem solve_inequality (x : ℝ) : x^2 - 3 * x - 10 < 0 ↔ -2 < x ∧ x < 5 := 
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1313_131329


namespace NUMINAMATH_GPT_cannot_tile_with_sphinxes_l1313_131373

def triangle_side_length : ℕ := 6
def small_triangles_count : ℕ := 36
def upward_triangles_count : ℕ := 21
def downward_triangles_count : ℕ := 15

theorem cannot_tile_with_sphinxes (n : ℕ) (small_triangles : ℕ) (upward : ℕ) (downward : ℕ) :
  n = triangle_side_length →
  small_triangles = small_triangles_count →
  upward = upward_triangles_count →
  downward = downward_triangles_count →
  (upward % 2 ≠ 0) ∨ (downward % 2 ≠ 0) →
  ¬ (upward + downward = small_triangles ∧
     ∀ k, (k * 6) ≤ small_triangles →
     ∃ u d, u + d = k * 6 ∧ u % 2 = 0 ∧ d % 2 = 0) := 
by
  intros
  sorry

end NUMINAMATH_GPT_cannot_tile_with_sphinxes_l1313_131373


namespace NUMINAMATH_GPT_odd_terms_in_expansion_l1313_131396

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (∃ k, k = 2) :=
sorry

end NUMINAMATH_GPT_odd_terms_in_expansion_l1313_131396


namespace NUMINAMATH_GPT_find_xyz_l1313_131380

theorem find_xyz
  (a b c x y z : ℂ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0)
  (h7 : a = (b + c) / (x - 3))
  (h8 : b = (a + c) / (y - 3))
  (h9 : c = (a + b) / (z - 3))
  (h10 : x * y + x * z + y * z = 10)
  (h11 : x + y + z = 6) :
  x * y * z = 10 :=
sorry

end NUMINAMATH_GPT_find_xyz_l1313_131380


namespace NUMINAMATH_GPT_Razorback_tshirt_shop_sales_l1313_131305

theorem Razorback_tshirt_shop_sales :
  let tshirt_price := 98
  let hat_price := 45
  let scarf_price := 60
  let tshirts_sold_arkansas := 42
  let hats_sold_arkansas := 32
  let scarves_sold_arkansas := 15
  (tshirts_sold_arkansas * tshirt_price + hats_sold_arkansas * hat_price + scarves_sold_arkansas * scarf_price) = 6456 :=
by
  sorry

end NUMINAMATH_GPT_Razorback_tshirt_shop_sales_l1313_131305


namespace NUMINAMATH_GPT_ratio_part_to_whole_l1313_131337

variable (N : ℝ)

theorem ratio_part_to_whole :
  (1 / 1) * (1 / 3) * (2 / 5) * N = 10 →
  0.4 * N = 120 →
  (10 / ((1 / 3) * (2 / 5) * N) = 1 / 4) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_ratio_part_to_whole_l1313_131337


namespace NUMINAMATH_GPT_cube_less_than_three_times_l1313_131379

theorem cube_less_than_three_times (x : ℤ) : x ^ 3 < 3 * x ↔ x = -3 ∨ x = -2 ∨ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_cube_less_than_three_times_l1313_131379


namespace NUMINAMATH_GPT_number_of_children_l1313_131386

def male_adults : ℕ := 60
def female_adults : ℕ := 60
def total_people : ℕ := 200

def total_adults : ℕ := male_adults + female_adults

theorem number_of_children : total_people - total_adults = 80 :=
by sorry

end NUMINAMATH_GPT_number_of_children_l1313_131386


namespace NUMINAMATH_GPT_intersection_x_coord_of_lines_l1313_131332

theorem intersection_x_coord_of_lines (k b : ℝ) (h : k ≠ b) :
  ∃ x : ℝ, (kx + b = bx + k) ∧ x = 1 :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_intersection_x_coord_of_lines_l1313_131332


namespace NUMINAMATH_GPT_women_at_dance_event_l1313_131325

theorem women_at_dance_event (men women : ℕ)
  (each_man_dances_with : ℕ)
  (each_woman_dances_with : ℕ)
  (total_men : men = 18)
  (dances_per_man : each_man_dances_with = 4)
  (dances_per_woman : each_woman_dances_with = 3)
  (total_dance_pairs : men * each_man_dances_with = 72) :
  women = 24 := 
  by {
    sorry
  }

end NUMINAMATH_GPT_women_at_dance_event_l1313_131325


namespace NUMINAMATH_GPT_equilateral_triangle_side_length_l1313_131381

theorem equilateral_triangle_side_length (side_length_of_square : ℕ) (h : side_length_of_square = 21) :
    let total_length_of_string := 4 * side_length_of_square
    let side_length_of_triangle := total_length_of_string / 3
    side_length_of_triangle = 28 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_side_length_l1313_131381


namespace NUMINAMATH_GPT_bake_sale_cookies_l1313_131394

theorem bake_sale_cookies (R O C : ℕ) (H1 : R = 42) (H2 : R = 6 * O) (H3 : R = 2 * C) : R + O + C = 70 := by
  sorry

end NUMINAMATH_GPT_bake_sale_cookies_l1313_131394


namespace NUMINAMATH_GPT_num_points_C_l1313_131387

theorem num_points_C (
  A B : ℝ × ℝ)
  (C : ℝ × ℝ) 
  (hA : A = (2, 2))
  (hB : B = (-1, -2))
  (hC : (C.1 - 3)^2 + (C.2 + 5)^2 = 36)
  (h_area : 1/2 * (abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))) = 5/2) :
  ∃ C1 C2 C3 : ℝ × ℝ,
    (C1.1 - 3)^2 + (C1.2 + 5)^2 = 36 ∧
    (C2.1 - 3)^2 + (C2.2 + 5)^2 = 36 ∧
    (C3.1 - 3)^2 + (C3.2 + 5)^2 = 36 ∧
    1/2 * (abs ((B.1 - A.1) * (C1.2 - A.2) - (B.2 - A.2) * (C1.1 - A.1))) = 5/2 ∧
    1/2 * (abs ((B.1 - A.1) * (C2.2 - A.2) - (B.2 - A.2) * (C2.1 - A.1))) = 5/2 ∧
    1/2 * (abs ((B.1 - A.1) * (C3.2 - A.2) - (B.2 - A.2) * (C3.1 - A.1))) = 5/2 ∧
    (C1 ≠ C2 ∧ C1 ≠ C3 ∧ C2 ≠ C3) :=
sorry

end NUMINAMATH_GPT_num_points_C_l1313_131387


namespace NUMINAMATH_GPT_sasha_remainder_l1313_131319

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end NUMINAMATH_GPT_sasha_remainder_l1313_131319


namespace NUMINAMATH_GPT_problem_gcd_polynomials_l1313_131344

theorem problem_gcd_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 7768 * k ∧ k % 2 = 0) :
  gcd (4 * b ^ 2 + 55 * b + 120) (3 * b + 12) = 12 :=
by
  sorry

end NUMINAMATH_GPT_problem_gcd_polynomials_l1313_131344


namespace NUMINAMATH_GPT_solution_l1313_131391

def mapping (x : ℝ) : ℝ := x^2

theorem solution (x : ℝ) : mapping x = 4 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_solution_l1313_131391


namespace NUMINAMATH_GPT_y_intercept_of_line_b_l1313_131383

-- Define the conditions
def line_parallel (m1 m2 : ℝ) : Prop := m1 = m2

def point_on_line (m b x y : ℝ) : Prop := y = m * x + b

-- Given conditions
variables (m b : ℝ)
variable (x₁ := 3)
variable (y₁ := -2)
axiom parallel_condition : line_parallel m (-3)
axiom point_condition : point_on_line m b x₁ y₁

-- Prove that the y-intercept b equals 7
theorem y_intercept_of_line_b : b = 7 :=
sorry

end NUMINAMATH_GPT_y_intercept_of_line_b_l1313_131383


namespace NUMINAMATH_GPT_translate_graph_downward_3_units_l1313_131362

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 1

theorem translate_graph_downward_3_units :
  ∀ x : ℝ, g x = f x - 3 :=
by
  sorry

end NUMINAMATH_GPT_translate_graph_downward_3_units_l1313_131362


namespace NUMINAMATH_GPT_find_x_l1313_131354

def hash_p (p : ℤ) (x : ℤ) : ℤ := 2 * p + x

def hash_of_hash_p (p : ℤ) (x : ℤ) : ℤ := 2 * hash_p p x + x

def triple_hash_p (p : ℤ) (x : ℤ) : ℤ := 2 * hash_of_hash_p p x + x

theorem find_x (p x : ℤ) (h : triple_hash_p p x = -4) (hp : p = 18) : x = -21 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1313_131354


namespace NUMINAMATH_GPT_train_time_l1313_131363

theorem train_time (T : ℕ) (D : ℝ) (h1 : D = 48 * (T / 60)) (h2 : D = 60 * (40 / 60)) : T = 50 :=
by
  sorry

end NUMINAMATH_GPT_train_time_l1313_131363


namespace NUMINAMATH_GPT_remainder_of_6x_mod_9_l1313_131314

theorem remainder_of_6x_mod_9 (x : ℕ) (h : x % 9 = 5) : (6 * x) % 9 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_6x_mod_9_l1313_131314
