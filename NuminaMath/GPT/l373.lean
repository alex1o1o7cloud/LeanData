import Mathlib

namespace NUMINAMATH_GPT_lina_collects_stickers_l373_37300

theorem lina_collects_stickers :
  let a := 3
  let d := 2
  let n := 10
  let a_n := a + (n - 1) * d
  let S_n := (n / 2) * (a + a_n)
  S_n = 120 :=
by
  sorry

end NUMINAMATH_GPT_lina_collects_stickers_l373_37300


namespace NUMINAMATH_GPT_determine_S_l373_37341

theorem determine_S :
  (∃ k : ℝ, (∀ S R T : ℝ, R = k * (S / T)) ∧ (∃ S R T : ℝ, R = 2 ∧ S = 6 ∧ T = 3 ∧ 2 = k * (6 / 3))) →
  (∀ S R T : ℝ, R = 8 ∧ T = 2 → S = 16) :=
by
  sorry

end NUMINAMATH_GPT_determine_S_l373_37341


namespace NUMINAMATH_GPT_frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l373_37389

-- Definitions based on the problem conditions
def machineA_first_class := 150
def machineA_total := 200
def machineB_first_class := 120
def machineB_total := 200
def total_products := machineA_total + machineB_total

-- Frequencies of first-class products
def frequency_machineA : ℚ := machineA_first_class / machineA_total
def frequency_machineB : ℚ := machineB_first_class / machineB_total

-- Values for chi-squared formula
def a := machineA_first_class
def b := machineA_total - machineA_first_class
def c := machineB_first_class
def d := machineB_total - machineB_first_class

-- Given formula for K^2
def K_squared : ℚ := (total_products * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Proof problem statements
theorem frequency_first_class_machineA_is_3_over_4 : frequency_machineA = 3 / 4 := by
  sorry

theorem frequency_first_class_machineB_is_3_over_5 : frequency_machineB = 3 / 5 := by
  sorry

theorem significant_quality_difference : K_squared > 6.635 := by
  sorry

end NUMINAMATH_GPT_frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l373_37389


namespace NUMINAMATH_GPT_calc_expression_l373_37368

theorem calc_expression :
  (- (2 / 5) : ℝ)^0 - (0.064 : ℝ)^(1/3) + 3^(Real.log (2 / 5) / Real.log 3) + Real.log 2 / Real.log 10 - Real.log (1 / 5) / Real.log 10 = 2 := 
by
  sorry

end NUMINAMATH_GPT_calc_expression_l373_37368


namespace NUMINAMATH_GPT_percentage_employees_four_years_or_more_l373_37395

theorem percentage_employees_four_years_or_more 
  (x : ℝ) 
  (less_than_one_year : ℝ := 6 * x)
  (one_to_two_years : ℝ := 4 * x)
  (two_to_three_years : ℝ := 7 * x)
  (three_to_four_years : ℝ := 3 * x)
  (four_to_five_years : ℝ := 3 * x)
  (five_to_six_years : ℝ := 1 * x)
  (six_to_seven_years : ℝ := 1 * x)
  (seven_to_eight_years : ℝ := 2 * x)
  (total_employees : ℝ := 27 * x)
  (employees_four_years_or_more : ℝ := 7 * x) : 
  (employees_four_years_or_more / total_employees) * 100 = 25.93 := 
by
  sorry

end NUMINAMATH_GPT_percentage_employees_four_years_or_more_l373_37395


namespace NUMINAMATH_GPT_tan_eq_tan_x2_sol_count_l373_37349

noncomputable def arctan1000 := Real.arctan 1000

theorem tan_eq_tan_x2_sol_count :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℝ, 
    0 ≤ x ∧ x ≤ arctan1000 ∧ Real.tan x = Real.tan (x^2) →
    ∃ k : ℕ, k < n ∧ x = Real.sqrt (k * Real.pi + x) :=
sorry

end NUMINAMATH_GPT_tan_eq_tan_x2_sol_count_l373_37349


namespace NUMINAMATH_GPT_third_number_pascals_triangle_61_numbers_l373_37312

theorem third_number_pascals_triangle_61_numbers : (Nat.choose 60 2) = 1770 := by
  sorry

end NUMINAMATH_GPT_third_number_pascals_triangle_61_numbers_l373_37312


namespace NUMINAMATH_GPT_sin_alpha_beta_value_l373_37370

theorem sin_alpha_beta_value (α β : ℝ) (h1 : 13 * Real.sin α + 5 * Real.cos β = 9) (h2 : 13 * Real.cos α + 5 * Real.sin β = 15) : 
  Real.sin (α + β) = 56 / 65 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_beta_value_l373_37370


namespace NUMINAMATH_GPT_village_population_equal_in_15_years_l373_37357

theorem village_population_equal_in_15_years :
  ∀ n : ℕ, (72000 - 1200 * n = 42000 + 800 * n) → n = 15 :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_village_population_equal_in_15_years_l373_37357


namespace NUMINAMATH_GPT_seven_people_different_rolls_l373_37351

def rolls_different (rolls : Fin 7 -> Fin 6) : Prop :=
  ∀ i : Fin 7, rolls i ≠ rolls ⟨(i + 1) % 7, sorry⟩

def probability_rolls_different : ℚ :=
  (625 : ℚ) / 2799

theorem seven_people_different_rolls (rolls : Fin 7 -> Fin 6) :
  (∃ rolls, rolls_different rolls) ->
  probability_rolls_different = 625 / 2799 :=
sorry

end NUMINAMATH_GPT_seven_people_different_rolls_l373_37351


namespace NUMINAMATH_GPT_range_of_m_l373_37308

theorem range_of_m (m x1 x2 y1 y2 : ℝ) (h1 : y1 = (1 + 2 * m) / x1) (h2 : y2 = (1 + 2 * m) / x2)
    (hx : x1 < 0 ∧ 0 < x2) (hy : y1 < y2) : m > -1 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l373_37308


namespace NUMINAMATH_GPT_sams_charge_per_sheet_is_1_5_l373_37335

variable (x : ℝ)
variable (a : ℝ) -- John's Photo World's charge per sheet
variable (b : ℝ) -- Sam's Picture Emporium's one-time sitting fee
variable (c : ℝ) -- John's Photo World's one-time sitting fee
variable (n : ℕ) -- Number of sheets

def johnsCost (n : ℕ) (a c : ℝ) := n * a + c
def samsCost (n : ℕ) (x b : ℝ) := n * x + b

theorem sams_charge_per_sheet_is_1_5 :
  ∀ (a b c : ℝ) (n : ℕ), a = 2.75 → b = 140 → c = 125 → n = 12 →
  johnsCost n a c = samsCost n x b → x = 1.50 := by
  intros a b c n ha hb hc hn h
  sorry

end NUMINAMATH_GPT_sams_charge_per_sheet_is_1_5_l373_37335


namespace NUMINAMATH_GPT_total_bees_is_25_l373_37324

def initial_bees : ℕ := 16
def additional_bees : ℕ := 9

theorem total_bees_is_25 : initial_bees + additional_bees = 25 := by
  sorry

end NUMINAMATH_GPT_total_bees_is_25_l373_37324


namespace NUMINAMATH_GPT_distance_between_stations_l373_37323

theorem distance_between_stations (x : ℕ) 
  (h1 : ∃ (x : ℕ), ∀ t : ℕ, (t * 16 = x ∧ t * 21 = x + 60)) :
  2 * x + 60 = 444 :=
by sorry

end NUMINAMATH_GPT_distance_between_stations_l373_37323


namespace NUMINAMATH_GPT_range_of_m_is_leq_3_l373_37345

noncomputable def is_range_of_m (m : ℝ) : Prop :=
  ∀ x : ℝ, 5^x + 3 > m

theorem range_of_m_is_leq_3 (m : ℝ) : is_range_of_m m ↔ m ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_is_leq_3_l373_37345


namespace NUMINAMATH_GPT_probability_less_than_8_rings_l373_37392

def P_10_ring : ℝ := 0.20
def P_9_ring : ℝ := 0.30
def P_8_ring : ℝ := 0.10

theorem probability_less_than_8_rings : 
  (1 - (P_10_ring + P_9_ring + P_8_ring)) = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_probability_less_than_8_rings_l373_37392


namespace NUMINAMATH_GPT_money_given_to_cashier_l373_37391

theorem money_given_to_cashier (regular_ticket_cost : ℕ) (discount : ℕ) 
  (age1 : ℕ) (age2 : ℕ) (change : ℕ) 
  (h1 : regular_ticket_cost = 109)
  (h2 : discount = 5)
  (h3 : age1 = 6)
  (h4 : age2 = 10)
  (h5 : change = 74)
  (h6 : age1 < 12)
  (h7 : age2 < 12) :
  regular_ticket_cost + regular_ticket_cost + (regular_ticket_cost - discount) + (regular_ticket_cost - discount) + change = 500 :=
by
  sorry

end NUMINAMATH_GPT_money_given_to_cashier_l373_37391


namespace NUMINAMATH_GPT_object_travel_distance_in_one_hour_l373_37318

/-- If an object travels at 3 feet per second, then it travels 10800 feet in one hour. -/
theorem object_travel_distance_in_one_hour
  (speed : ℕ) (seconds_in_minute : ℕ) (minutes_in_hour : ℕ)
  (h_speed : speed = 3)
  (h_seconds_in_minute : seconds_in_minute = 60)
  (h_minutes_in_hour : minutes_in_hour = 60) :
  (speed * (seconds_in_minute * minutes_in_hour) = 10800) :=
by
  sorry

end NUMINAMATH_GPT_object_travel_distance_in_one_hour_l373_37318


namespace NUMINAMATH_GPT_find_k_l373_37328

theorem find_k {k : ℝ} (h : (∃ α β : ℝ, α ≠ 0 ∧ β ≠ 0 ∧ α / β = 3 / 1 ∧ α + β = -10 ∧ α * β = k)) : k = 18.75 :=
sorry

end NUMINAMATH_GPT_find_k_l373_37328


namespace NUMINAMATH_GPT_original_cost_price_l373_37320

theorem original_cost_price (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1275) 
  (h2 : loss_percentage = 15) 
  (h3 : SP = (1 - loss_percentage / 100) * C) : 
  C = 1500 := 
by 
  sorry

end NUMINAMATH_GPT_original_cost_price_l373_37320


namespace NUMINAMATH_GPT_solution_is_permutations_l373_37398

noncomputable def solve_system (x y z : ℤ) : Prop :=
  x^2 = y * z + 1 ∧ y^2 = z * x + 1 ∧ z^2 = x * y + 1

theorem solution_is_permutations (x y z : ℤ) :
  solve_system x y z ↔ (x, y, z) = (1, 0, -1) ∨ (x, y, z) = (1, -1, 0) ∨ (x, y, z) = (0, 1, -1) ∨ (x, y, z) = (0, -1, 1) ∨ (x, y, z) = (-1, 1, 0) ∨ (x, y, z) = (-1, 0, 1) :=
by sorry

end NUMINAMATH_GPT_solution_is_permutations_l373_37398


namespace NUMINAMATH_GPT_bike_ride_distance_l373_37340

-- Definitions for conditions from a)
def speed_out := 24 -- miles per hour
def speed_back := 18 -- miles per hour
def total_time := 7 -- hours

-- Problem statement for the proof problem
theorem bike_ride_distance :
  ∃ (D : ℝ), (D / speed_out) + (D / speed_back) = total_time ∧ 2 * D = 144 :=
by {
  sorry
}

end NUMINAMATH_GPT_bike_ride_distance_l373_37340


namespace NUMINAMATH_GPT_no_discrepancy_l373_37310

-- Definitions based on the conditions
def t1_hours : ℝ := 1.5 -- time taken clockwise in hours
def t2_minutes : ℝ := 90 -- time taken counterclockwise in minutes

-- Lean statement to prove the equivalence
theorem no_discrepancy : t1_hours * 60 = t2_minutes :=
by sorry

end NUMINAMATH_GPT_no_discrepancy_l373_37310


namespace NUMINAMATH_GPT_worker_usual_time_l373_37347

theorem worker_usual_time (T : ℝ) (S : ℝ) (h₀ : S > 0) (h₁ : (4 / 5) * S * (T + 10) = S * T) : T = 40 :=
sorry

end NUMINAMATH_GPT_worker_usual_time_l373_37347


namespace NUMINAMATH_GPT_number_of_ordered_triples_l373_37394

theorem number_of_ordered_triples :
  let b := 2023
  let n := (b ^ 2)
  ∀ (a c : ℕ), a * c = n ∧ a ≤ b ∧ b ≤ c → (∃ (k : ℕ), k = 7) :=
by
  sorry

end NUMINAMATH_GPT_number_of_ordered_triples_l373_37394


namespace NUMINAMATH_GPT_product_xyz_l373_37301

theorem product_xyz (x y z : ℝ) (h1 : x = y) (h2 : x = 2 * z) (h3 : x = 7.999999999999999) :
    x * y * z = 255.9999999999998 := by
  sorry

end NUMINAMATH_GPT_product_xyz_l373_37301


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l373_37393

variable {x y : ℝ}

theorem ratio_of_x_to_y (h1 : (3 * x - 2 * y) / (2 * x + 3 * y) = 5 / 4) (h2 : x + y = 5) : x / y = 23 / 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_ratio_of_x_to_y_l373_37393


namespace NUMINAMATH_GPT_theresa_crayons_count_l373_37365

noncomputable def crayons_teresa (initial_teresa_crayons : Nat) 
                                 (initial_janice_crayons : Nat) 
                                 (shared_with_nancy : Nat)
                                 (given_to_mark : Nat)
                                 (received_from_nancy : Nat) : Nat := 
  initial_teresa_crayons + received_from_nancy

theorem theresa_crayons_count : crayons_teresa 32 12 (12 / 2) 3 8 = 40 := by
  -- Given: Theresa initially has 32 crayons.
  -- Janice initially has 12 crayons.
  -- Janice shares half of her crayons with Nancy: 12 / 2 = 6 crayons.
  -- Janice gives 3 crayons to Mark.
  -- Theresa receives 8 crayons from Nancy.
  -- Therefore: Theresa will have 32 + 8 = 40 crayons.
  sorry

end NUMINAMATH_GPT_theresa_crayons_count_l373_37365


namespace NUMINAMATH_GPT_inverse_proportional_t_no_linear_function_2k_times_quadratic_function_5_times_l373_37363

-- Proof Problem 1
theorem inverse_proportional_t (t : ℝ) (h1 : 1 ≤ t ∧ t ≤ 2023) : t = 1 :=
sorry

-- Proof Problem 2
theorem no_linear_function_2k_times (k : ℝ) (h_pos : 0 < k) : ¬ ∃ a b : ℝ, (a < b) ∧ (∀ x, a ≤ x ∧ x ≤ b → (2 * k * a ≤ k * x + 2 ∧ k * x + 2 ≤ 2 * k * b)) :=
sorry

-- Proof Problem 3
theorem quadratic_function_5_times (a b : ℝ) (h_ab : a < b) (h_quad : ∀ x, a ≤ x ∧ x ≤ b → (5 * a ≤ x^2 - 4 * x - 7 ∧ x^2 - 4 * x - 7 ≤ 5 * b)) :
  (a = -2 ∧ b = 1) ∨ (a = -(11/5) ∧ b = (9 + Real.sqrt 109) / 2) :=
sorry

end NUMINAMATH_GPT_inverse_proportional_t_no_linear_function_2k_times_quadratic_function_5_times_l373_37363


namespace NUMINAMATH_GPT_martha_cards_l373_37399

theorem martha_cards :
  let initial_cards := 3
  let emily_cards := 25
  let alex_cards := 43
  let jenny_cards := 58
  let sam_cards := 14
  initial_cards + emily_cards + alex_cards + jenny_cards - sam_cards = 115 := 
by
  sorry

end NUMINAMATH_GPT_martha_cards_l373_37399


namespace NUMINAMATH_GPT_robotics_club_neither_l373_37361

theorem robotics_club_neither (total_students cs_students e_students both_students : ℕ)
  (h1 : total_students = 80)
  (h2 : cs_students = 52)
  (h3 : e_students = 45)
  (h4 : both_students = 32) :
  total_students - (cs_students - both_students + e_students - both_students + both_students) = 15 :=
by
  sorry

end NUMINAMATH_GPT_robotics_club_neither_l373_37361


namespace NUMINAMATH_GPT_fat_rings_per_group_l373_37333

theorem fat_rings_per_group (F : ℕ)
  (h1 : ∀ F, (70 * (F + 4)) = (40 * (F + 4)) + 180)
  : F = 2 :=
sorry

end NUMINAMATH_GPT_fat_rings_per_group_l373_37333


namespace NUMINAMATH_GPT_selling_prices_l373_37356

theorem selling_prices {x y : ℝ} (h1 : y - x = 10) (h2 : (y - 5) - 1.10 * x = 1) :
  x = 40 ∧ y = 50 := by
  sorry

end NUMINAMATH_GPT_selling_prices_l373_37356


namespace NUMINAMATH_GPT_total_passengers_l373_37307

theorem total_passengers (P : ℕ)
  (h1 : P / 12 + P / 8 + P / 3 + P / 6 + 35 = P) : 
  P = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_passengers_l373_37307


namespace NUMINAMATH_GPT_custom_op_4_3_l373_37329

-- Define the custom operation a * b
def custom_op (a b : ℤ) : ℤ := a^2 + a * b - b^2

-- State the theorem to be proven
theorem custom_op_4_3 : custom_op 4 3 = 19 := 
by
sorry

end NUMINAMATH_GPT_custom_op_4_3_l373_37329


namespace NUMINAMATH_GPT_not_right_angled_triangle_l373_37355

theorem not_right_angled_triangle 
  (m n : ℝ) 
  (h1 : m > n) 
  (h2 : n > 0)
  : ¬ (m^2 + n^2)^2 = (mn)^2 + (m^2 - n^2)^2 :=
sorry

end NUMINAMATH_GPT_not_right_angled_triangle_l373_37355


namespace NUMINAMATH_GPT_michael_cleanings_total_l373_37350

theorem michael_cleanings_total (baths_per_week : ℕ) (showers_per_week : ℕ) (weeks_in_year : ℕ) 
  (h_baths : baths_per_week = 2) (h_showers : showers_per_week = 1) (h_weeks : weeks_in_year = 52) :
  (baths_per_week + showers_per_week) * weeks_in_year = 156 :=
by 
  -- Omitting proof as instructed.
  sorry

end NUMINAMATH_GPT_michael_cleanings_total_l373_37350


namespace NUMINAMATH_GPT_red_cards_count_l373_37313

theorem red_cards_count (R B : ℕ) (h1 : R + B = 20) (h2 : 3 * R + 5 * B = 84) : R = 8 :=
sorry

end NUMINAMATH_GPT_red_cards_count_l373_37313


namespace NUMINAMATH_GPT_probability_age_21_to_30_l373_37376

theorem probability_age_21_to_30 : 
  let total_people := 160 
  let people_10_to_20 := 40
  let people_21_to_30 := 70
  let people_31_to_40 := 30
  let people_41_to_50 := 20
  (people_21_to_30 / total_people : ℚ) = 7 / 16 := by
  sorry

end NUMINAMATH_GPT_probability_age_21_to_30_l373_37376


namespace NUMINAMATH_GPT_sum_odd_digits_from_1_to_200_l373_37326

/-- Function to compute the sum of odd digits of a number -/
def odd_digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.filter (fun d => d % 2 = 1) |>.sum

/-- Statement of the problem to prove the sum of the odd digits of numbers from 1 to 200 is 1000 -/
theorem sum_odd_digits_from_1_to_200 : (Finset.range 200).sum odd_digit_sum = 1000 := 
  sorry

end NUMINAMATH_GPT_sum_odd_digits_from_1_to_200_l373_37326


namespace NUMINAMATH_GPT_quadrilateral_diagonals_l373_37373

theorem quadrilateral_diagonals (a b c d e f : ℝ) 
  (hac : a > c) 
  (hbd : b ≥ d) 
  (hapc : a = c) 
  (hdiag1 : e^2 = (a - b)^2 + b^2) 
  (hdiag2 : f^2 = (c + b)^2 + b^2) :
  e^4 - f^4 = (a + c) / (a - c) * (d^2 * (2 * a * c + d^2) - b^2 * (2 * a * c + b^2)) :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_diagonals_l373_37373


namespace NUMINAMATH_GPT_directrix_of_parabola_l373_37390

theorem directrix_of_parabola (y : ℝ) : 
  (∃ y : ℝ, x = 1) ↔ (x = (1 / 4 : ℝ) * y^2) := 
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l373_37390


namespace NUMINAMATH_GPT_probability_individual_selected_l373_37380

/-- Given a population of 8 individuals, the probability that each 
individual is selected in a simple random sample of size 4 is 1/2. -/
theorem probability_individual_selected :
  let population_size := 8
  let sample_size := 4
  let probability := sample_size / population_size
  probability = (1 : ℚ) / 2 :=
by
  let population_size := 8
  let sample_size := 4
  let probability := sample_size / population_size
  sorry

end NUMINAMATH_GPT_probability_individual_selected_l373_37380


namespace NUMINAMATH_GPT_probability_draw_l373_37311

theorem probability_draw (pA_win pA_not_lose : ℝ) (h1 : pA_win = 0.3) (h2 : pA_not_lose = 0.8) :
  pA_not_lose - pA_win = 0.5 :=
by 
  sorry

end NUMINAMATH_GPT_probability_draw_l373_37311


namespace NUMINAMATH_GPT_last_digit_of_large_prime_l373_37358

theorem last_digit_of_large_prime : 
  (859433 = 214858 * 4 + 1) → 
  (∃ d, (2 ^ 859433 - 1) % 10 = d ∧ d = 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_last_digit_of_large_prime_l373_37358


namespace NUMINAMATH_GPT_speed_of_second_train_l373_37371

/-- Given:
1. The first train has a length of 220 meters.
2. The speed of the first train is 120 kilometers per hour.
3. The time taken to cross each other is 9 seconds.
4. The length of the second train is 280.04 meters.

Prove the speed of the second train is 80 kilometers per hour. -/
theorem speed_of_second_train
    (len_first_train : ℝ := 220)
    (speed_first_train_kmph : ℝ := 120)
    (time_to_cross : ℝ := 9)
    (len_second_train : ℝ := 280.04) 
  : (len_first_train / time_to_cross + len_second_train / time_to_cross - (speed_first_train_kmph * 1000 / 3600)) * (3600 / 1000) = 80 := 
by
  sorry

end NUMINAMATH_GPT_speed_of_second_train_l373_37371


namespace NUMINAMATH_GPT_correct_quadratic_equation_l373_37302

def is_quadratic_with_one_variable (eq : String) : Prop :=
  eq = "x^2 + 1 = 0"

theorem correct_quadratic_equation :
  is_quadratic_with_one_variable "x^2 + 1 = 0" :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_quadratic_equation_l373_37302


namespace NUMINAMATH_GPT_length_of_plot_is_60_l373_37342

noncomputable def plot_length (b : ℝ) : ℝ :=
  b + 20

noncomputable def plot_perimeter (b : ℝ) : ℝ :=
  2 * (plot_length b + b)

noncomputable def plot_cost_eq (b : ℝ) : Prop :=
  26.50 * plot_perimeter b = 5300

theorem length_of_plot_is_60 : ∃ b : ℝ, plot_cost_eq b ∧ plot_length b = 60 :=
sorry

end NUMINAMATH_GPT_length_of_plot_is_60_l373_37342


namespace NUMINAMATH_GPT_gap_between_rails_should_be_12_24_mm_l373_37303

noncomputable def initial_length : ℝ := 15
noncomputable def temperature_initial : ℝ := -8
noncomputable def temperature_max : ℝ := 60
noncomputable def expansion_coefficient : ℝ := 0.000012
noncomputable def change_in_temperature : ℝ := temperature_max - temperature_initial
noncomputable def final_length : ℝ := initial_length * (1 + expansion_coefficient * change_in_temperature)
noncomputable def gap : ℝ := (final_length - initial_length) * 1000  -- converted to mm

theorem gap_between_rails_should_be_12_24_mm
  : gap = 12.24 := by
  sorry

end NUMINAMATH_GPT_gap_between_rails_should_be_12_24_mm_l373_37303


namespace NUMINAMATH_GPT_parabola_tangent_parameter_l373_37366

theorem parabola_tangent_parameter (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hp : p ≠ 0) :
  ∃ p : ℝ, (∀ y, y^2 + (2 * p * b / a) * y + (2 * p * c^2 / a) = 0) ↔ (p = 2 * a * c^2 / b^2) := 
by
  sorry

end NUMINAMATH_GPT_parabola_tangent_parameter_l373_37366


namespace NUMINAMATH_GPT_sqrt_simplification_l373_37379

theorem sqrt_simplification : Real.sqrt 360000 = 600 :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_simplification_l373_37379


namespace NUMINAMATH_GPT_number_of_pints_of_paint_l373_37306

-- Statement of the problem
theorem number_of_pints_of_paint (A B : ℝ) (N : ℕ) 
  (large_cube_paint : ℝ) (hA : A = 4) (hB : B = 2) (hN : N = 125) 
  (large_cube_paint_condition : large_cube_paint = 1) : 
  (N * (B / A) ^ 2 * large_cube_paint = 31.25) :=
by {
  -- Given the conditions
  sorry
}

end NUMINAMATH_GPT_number_of_pints_of_paint_l373_37306


namespace NUMINAMATH_GPT_factor_transformation_option_C_l373_37346

theorem factor_transformation_option_C (y : ℝ) : 
  4 * y^2 - 4 * y + 1 = (2 * y - 1)^2 :=
sorry

end NUMINAMATH_GPT_factor_transformation_option_C_l373_37346


namespace NUMINAMATH_GPT_trig_identity_l373_37343

theorem trig_identity : 
  (2 * Real.sin (80 * Real.pi / 180) - Real.sin (20 * Real.pi / 180)) / Real.cos (20 * Real.pi / 180) = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_trig_identity_l373_37343


namespace NUMINAMATH_GPT_find_variable_l373_37359

def expand : ℤ → ℤ := 3*2*6
    
theorem find_variable (a n some_variable : ℤ) (h : (3 - 7 + a = 3)):
  some_variable = -17 :=
sorry

end NUMINAMATH_GPT_find_variable_l373_37359


namespace NUMINAMATH_GPT_quadratic_solution_identity_l373_37316

theorem quadratic_solution_identity (a b : ℤ) (h : (1 : ℤ)^2 + a * 1 + 2 * b = 0) : 2 * a + 4 * b = -2 := by
  sorry

end NUMINAMATH_GPT_quadratic_solution_identity_l373_37316


namespace NUMINAMATH_GPT_problem_statement_l373_37344

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (a + 1 / a) ^ 2 + (b + 1 / b) ^ 2 ≥ 25 / 2 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l373_37344


namespace NUMINAMATH_GPT_polarEquationOfCircleCenter1_1Radius1_l373_37334

noncomputable def circleEquationInPolarCoordinates (θ : ℝ) : ℝ := 2 * Real.cos (θ - 1)

theorem polarEquationOfCircleCenter1_1Radius1 (ρ θ : ℝ) 
  (h : Real.sqrt ((ρ * Real.cos θ - Real.cos 1)^2 + (ρ * Real.sin θ - Real.sin 1)^2) = 1) :
  ρ = circleEquationInPolarCoordinates θ :=
by sorry

end NUMINAMATH_GPT_polarEquationOfCircleCenter1_1Radius1_l373_37334


namespace NUMINAMATH_GPT_ratio_wrong_to_correct_l373_37315

theorem ratio_wrong_to_correct (total_sums correct_sums : ℕ) 
  (h1 : total_sums = 36) (h2 : correct_sums = 12) : 
  (total_sums - correct_sums) / correct_sums = 2 :=
by {
  -- Proof will go here
  sorry
}

end NUMINAMATH_GPT_ratio_wrong_to_correct_l373_37315


namespace NUMINAMATH_GPT_poly_has_two_distinct_negative_real_roots_l373_37317

-- Definition of the polynomial equation
def poly_eq (p x : ℝ) : Prop :=
  x^4 + 4*p*x^3 + 2*x^2 + 4*p*x + 1 = 0

-- Theorem statement that needs to be proved
theorem poly_has_two_distinct_negative_real_roots (p : ℝ) :
  p > 1 → ∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ poly_eq p x1 ∧ poly_eq p x2 :=
by
  sorry

end NUMINAMATH_GPT_poly_has_two_distinct_negative_real_roots_l373_37317


namespace NUMINAMATH_GPT_geometric_representation_l373_37352

variables (a : ℝ)

-- Definition of the area of the figure
def total_area := a^2 + 1.5 * a

-- Definition of the perimeter of the figure
def total_perimeter := 4 * a + 3

theorem geometric_representation :
  total_area a = a^2 + 1.5 * a ∧ total_perimeter a = 4 * a + 3 :=
by
  exact ⟨rfl, rfl⟩

end NUMINAMATH_GPT_geometric_representation_l373_37352


namespace NUMINAMATH_GPT_total_earnings_of_a_b_c_l373_37369

theorem total_earnings_of_a_b_c 
  (days_a days_b days_c : ℕ)
  (ratio_a ratio_b ratio_c : ℕ)
  (wage_c : ℕ) 
  (h_ratio : ratio_a * wage_c = 3 * (3 + 4 + 5))
  (h_ratio_a_b : ratio_b = 4 * wage_c / 5 * ratio_a / 60)
  (h_ratio_b_c : ratio_b = 4 * wage_c / 5 * ratio_c / 60):
  (ratio_a * days_a + ratio_b * days_b + ratio_c * days_c) = 1480 := 
  by
    sorry

end NUMINAMATH_GPT_total_earnings_of_a_b_c_l373_37369


namespace NUMINAMATH_GPT_old_selling_price_l373_37314

theorem old_selling_price (C : ℝ) 
  (h1 : C + 0.15 * C = 92) :
  C + 0.10 * C = 88 :=
by
  sorry

end NUMINAMATH_GPT_old_selling_price_l373_37314


namespace NUMINAMATH_GPT_circle_diameter_given_area_l373_37386

theorem circle_diameter_given_area : 
  (∃ (r : ℝ), 81 * Real.pi = Real.pi * r^2 ∧ 2 * r = d) → d = 18 := by
  sorry

end NUMINAMATH_GPT_circle_diameter_given_area_l373_37386


namespace NUMINAMATH_GPT_triangle_integer_solutions_l373_37367

theorem triangle_integer_solutions (x : ℕ) (h1 : 13 < x) (h2 : x < 43) : 
  ∃ (n : ℕ), n = 29 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_integer_solutions_l373_37367


namespace NUMINAMATH_GPT_max_sum_l373_37339

open Real

theorem max_sum (a b c : ℝ) (h : a^2 + (b^2) / 4 + (c^2) / 9 = 1) : a + b + c ≤ sqrt 14 :=
sorry

end NUMINAMATH_GPT_max_sum_l373_37339


namespace NUMINAMATH_GPT_solve_inequality_l373_37330

noncomputable def rational_inequality_solution (x : ℝ) : Prop :=
  3 - (x^2 - 4 * x - 5) / (3 * x + 2) > 1

theorem solve_inequality (x : ℝ) :
  rational_inequality_solution x ↔ (x > -2 / 3 ∧ x < 9) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l373_37330


namespace NUMINAMATH_GPT_min_value_of_quadratic_expression_l373_37375

theorem min_value_of_quadratic_expression : ∃ x : ℝ, ∀ y : ℝ, y = x^2 + 12*x + 9 → y ≥ -27 :=
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_expression_l373_37375


namespace NUMINAMATH_GPT_find_three_digit_number_l373_37377

theorem find_three_digit_number (A B C : ℕ) (h1 : A + B + C = 10) (h2 : B = A + C) (h3 : 100 * C + 10 * B + A = 100 * A + 10 * B + C + 99) : 100 * A + 10 * B + C = 253 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_three_digit_number_l373_37377


namespace NUMINAMATH_GPT_factorize_def_l373_37319

def factorize_polynomial (p q r : Polynomial ℝ) : Prop :=
  p = q * r

theorem factorize_def (p q r : Polynomial ℝ) :
  factorize_polynomial p q r → p = q * r :=
  sorry

end NUMINAMATH_GPT_factorize_def_l373_37319


namespace NUMINAMATH_GPT_division_by_ab_plus_one_is_perfect_square_l373_37381

theorem division_by_ab_plus_one_is_perfect_square
    (a b : ℕ) (h : 0 < a ∧ 0 < b)
    (hab : (ab + 1) ∣ (a^2 + b^2)) :
    ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) := 
sorry

end NUMINAMATH_GPT_division_by_ab_plus_one_is_perfect_square_l373_37381


namespace NUMINAMATH_GPT_x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5_l373_37364

theorem x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5 :
  (∀ x : ℝ, x > 5 → x > 3) ∧ ¬(∀ x : ℝ, x > 3 → x > 5) :=
by 
  -- Prove implications with provided conditions
  sorry

end NUMINAMATH_GPT_x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5_l373_37364


namespace NUMINAMATH_GPT_min_value_frac_l373_37383

variable (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1)

theorem min_value_frac : (1 / a + 4 / b) = 9 :=
by sorry

end NUMINAMATH_GPT_min_value_frac_l373_37383


namespace NUMINAMATH_GPT_elephant_entry_rate_l373_37374

-- Define the variables and constants
def initial_elephants : ℕ := 30000
def exit_rate : ℕ := 2880
def exit_time : ℕ := 4
def enter_time : ℕ := 7
def final_elephants : ℕ := 28980

-- Prove the rate of new elephants entering the park
theorem elephant_entry_rate :
  (final_elephants - (initial_elephants - exit_rate * exit_time)) / enter_time = 1500 :=
by
  sorry -- placeholder for the proof

end NUMINAMATH_GPT_elephant_entry_rate_l373_37374


namespace NUMINAMATH_GPT_land_for_cattle_l373_37378

-- Define the conditions as Lean definitions
def total_land : ℕ := 150
def house_and_machinery : ℕ := 25
def future_expansion : ℕ := 15
def crop_production : ℕ := 70

-- Statement to prove
theorem land_for_cattle : total_land - (house_and_machinery + future_expansion + crop_production) = 40 :=
by
  sorry

end NUMINAMATH_GPT_land_for_cattle_l373_37378


namespace NUMINAMATH_GPT_smallest_x_y_z_sum_l373_37396

theorem smallest_x_y_z_sum :
  ∃ x y z : ℝ, x + 3*y + 6*z = 1 ∧ x*y + 2*x*z + 6*y*z = -8 ∧ x*y*z = 2 ∧ x + y + z = -(8/3) := 
sorry

end NUMINAMATH_GPT_smallest_x_y_z_sum_l373_37396


namespace NUMINAMATH_GPT_smallest_solution_l373_37322

theorem smallest_solution (x : ℝ) (h : x^2 + 10 * x - 24 = 0) : x = -12 :=
sorry

end NUMINAMATH_GPT_smallest_solution_l373_37322


namespace NUMINAMATH_GPT_complement_union_correct_l373_37332

open Set

theorem complement_union_correct :
  let P : Set ℕ := { x | x * (x - 3) ≥ 0 }
  let Q : Set ℕ := {2, 4}
  (compl P) ∪ Q = {1, 2, 4} :=
by
  let P : Set ℕ := { x | x * (x - 3) ≥ 0 }
  let Q : Set ℕ := {2, 4}
  have h : (compl P) ∪ Q = {1, 2, 4} := sorry
  exact h

end NUMINAMATH_GPT_complement_union_correct_l373_37332


namespace NUMINAMATH_GPT_ball_arrangement_l373_37348

theorem ball_arrangement : ∃ (n : ℕ), n = 120 ∧
  (∀ (ball_count : ℕ), ball_count = 20 → ∃ (box1 box2 box3 : ℕ), 
    box1 ≥ 1 ∧ box2 ≥ 2 ∧ box3 ≥ 3 ∧ box1 + box2 + box3 = ball_count) :=
by
  sorry

end NUMINAMATH_GPT_ball_arrangement_l373_37348


namespace NUMINAMATH_GPT_sparrow_swallow_equations_l373_37372

theorem sparrow_swallow_equations (x y : ℝ) : 
  (5 * x + 6 * y = 16) ∧ (4 * x + y = 5 * y + x) :=
  sorry

end NUMINAMATH_GPT_sparrow_swallow_equations_l373_37372


namespace NUMINAMATH_GPT_area_of_inscribed_rectangle_not_square_area_of_inscribed_rectangle_is_square_l373_37385

theorem area_of_inscribed_rectangle_not_square (s : ℝ) : 
  (s > 0) ∧ (s < 1 / 2) :=
sorry

theorem area_of_inscribed_rectangle_is_square (s : ℝ) : 
  (s >= 1 / 2) ∧ (s < 1) :=
sorry

end NUMINAMATH_GPT_area_of_inscribed_rectangle_not_square_area_of_inscribed_rectangle_is_square_l373_37385


namespace NUMINAMATH_GPT_trig_identity_l373_37353

open Real

theorem trig_identity (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 6) (h : sin α ^ 6 + cos α ^ 6 = 7 / 12) : 1998 * cos α = 333 * Real.sqrt 30 :=
sorry

end NUMINAMATH_GPT_trig_identity_l373_37353


namespace NUMINAMATH_GPT_ellipse_m_gt_5_l373_37387

theorem ellipse_m_gt_5 (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2) → m > 5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_ellipse_m_gt_5_l373_37387


namespace NUMINAMATH_GPT_car_speed_return_trip_l373_37338

noncomputable def speed_return_trip (d : ℕ) (v_ab : ℕ) (v_avg : ℕ) : ℕ := 
  (2 * d * v_avg) / (2 * v_avg - v_ab)

theorem car_speed_return_trip :
  let d := 180
  let v_ab := 90
  let v_avg := 60
  speed_return_trip d v_ab v_avg = 45 :=
by
  simp [speed_return_trip]
  sorry

end NUMINAMATH_GPT_car_speed_return_trip_l373_37338


namespace NUMINAMATH_GPT_relationship_between_P_and_Q_l373_37384

def P (x : ℝ) : Prop := x < 1
def Q (x : ℝ) : Prop := (x + 2) * (x - 1) < 0

theorem relationship_between_P_and_Q : 
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬ Q x) :=
sorry

end NUMINAMATH_GPT_relationship_between_P_and_Q_l373_37384


namespace NUMINAMATH_GPT_solve_rings_l373_37304

variable (B : ℝ) (S : ℝ)

def conditions := (S = (5/8) * (Real.sqrt B)) ∧ (S + B = 52)

theorem solve_rings : conditions B S → (S + B = 52) := by
  intros h
  sorry

end NUMINAMATH_GPT_solve_rings_l373_37304


namespace NUMINAMATH_GPT_Ms_Hatcher_total_students_l373_37331

noncomputable def number_of_students (third_graders fourth_graders fifth_graders sixth_graders : ℕ) : ℕ :=
  third_graders + fourth_graders + fifth_graders + sixth_graders

theorem Ms_Hatcher_total_students (third_graders fourth_graders fifth_graders sixth_graders : ℕ) 
  (h1 : third_graders = 20)
  (h2 : fourth_graders = 2 * third_graders) 
  (h3 : fifth_graders = third_graders / 2) 
  (h4 : sixth_graders = 3 * (third_graders + fourth_graders) / 4) : 
  number_of_students third_graders fourth_graders fifth_graders sixth_graders = 115 :=
by
  sorry

end NUMINAMATH_GPT_Ms_Hatcher_total_students_l373_37331


namespace NUMINAMATH_GPT_problem_1_problem_2_l373_37309

theorem problem_1 (n : ℕ) (h : n > 0) (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, (n > 0) → 
    (∃ α β, α + β = β * α + 1 ∧ 
            α * β = 1 / a n ∧ 
            a n * α^2 - a (n+1) * α + 1 = 0 ∧ 
            a n * β^2 - a (n+1) * β + 1 = 0)) :
  a (n + 1) = a n + 1 := sorry

theorem problem_2 (n : ℕ) (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n, (n > 0) → a (n+1) = a n + 1) :
  a n = n := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l373_37309


namespace NUMINAMATH_GPT_correct_sum_is_1826_l373_37388

-- Define the four-digit number representation
def four_digit (A B C D : ℕ) := 1000 * A + 100 * B + 10 * C + D

-- Condition: Yoongi confused the units digit (9 as 6)
-- The incorrect number Yoongi used
def incorrect_number (A B C : ℕ) := four_digit A B C 6

-- The correct number
def correct_number (A B C : ℕ) := four_digit A B C 9

-- The sum obtained by Yoongi
def yoongi_sum (A B C : ℕ) := incorrect_number A B C + 57

-- The correct sum 
def correct_sum (A B C : ℕ) := correct_number A B C + 57

-- Condition: Yoongi's sum is 1823
axiom yoongi_sum_is_1823 (A B C: ℕ) : yoongi_sum A B C = 1823

-- Proof Problem: Prove that the correct sum is 1826
theorem correct_sum_is_1826 (A B C : ℕ) : correct_sum A B C = 1826 := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_correct_sum_is_1826_l373_37388


namespace NUMINAMATH_GPT_circle_passing_points_l373_37336

theorem circle_passing_points :
  ∃ (D E F : ℝ), 
    (25 + 1 + 5 * D + E + F = 0) ∧ 
    (36 + 6 * D + F = 0) ∧ 
    (1 + 1 - D + E + F = 0) ∧ 
    (∀ x y : ℝ, (x, y) = (5, 1) ∨ (x, y) = (6, 0) ∨ (x, y) = (-1, 1) → x^2 + y^2 + D * x + E * y + F = 0) → 
  x^2 + y^2 - 4 * x + 6 * y - 12 = 0 :=
by
  sorry

end NUMINAMATH_GPT_circle_passing_points_l373_37336


namespace NUMINAMATH_GPT_triangle_angle_l373_37360

theorem triangle_angle (A B C : ℝ) (h1 : A - C = B) (h2 : A + B + C = 180) : A = 90 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_l373_37360


namespace NUMINAMATH_GPT_fraction_is_correct_l373_37337

def f (x : ℕ) : ℕ := 3 * x + 2
def g (x : ℕ) : ℕ := 2 * x - 3

theorem fraction_is_correct : (f (g (f 3))) / (g (f (g 3))) = 59 / 19 :=
by
  sorry

end NUMINAMATH_GPT_fraction_is_correct_l373_37337


namespace NUMINAMATH_GPT_ratio_PM_MQ_eq_1_l373_37362

theorem ratio_PM_MQ_eq_1
  (A B C D E M P Q : ℝ × ℝ)
  (square_side : ℝ)
  (h_square_side : square_side = 15)
  (hA : A = (0, square_side))
  (hB : B = (square_side, square_side))
  (hC : C = (square_side, 0))
  (hD : D = (0, 0))
  (hE : E = (8, 0))
  (hM : M = ((A.1 + E.1) / 2, (A.2 + E.2) / 2))
  (h_slope_AE : E.2 - A.2 = (E.1 - A.1) * -15 / 8)
  (h_P_on_AD : P.2 = 15)
  (h_Q_on_BC : Q.2 = 0)
  (h_PM_len : dist M P = dist M Q) :
  dist P M = dist M Q :=
by sorry

end NUMINAMATH_GPT_ratio_PM_MQ_eq_1_l373_37362


namespace NUMINAMATH_GPT_red_before_green_probability_l373_37325

open Classical

noncomputable def probability_red_before_green (total_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ) : ℚ :=
  let total_arrangements := (Nat.choose (total_chips - 1) green_chips)
  let favorable_arrangements := Nat.choose (total_chips - red_chips - 1) (green_chips - 1)
  favorable_arrangements / total_arrangements

theorem red_before_green_probability :
  probability_red_before_green 8 4 3 = 3 / 7 :=
sorry

end NUMINAMATH_GPT_red_before_green_probability_l373_37325


namespace NUMINAMATH_GPT_cleaning_time_together_l373_37321

theorem cleaning_time_together (t : ℝ) (h_t : 3 = t / 3) (h_john_time : 6 = 6) : 
  (5 / (1 / 6 + 1 / 9)) = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_cleaning_time_together_l373_37321


namespace NUMINAMATH_GPT_other_train_length_l373_37327

noncomputable def length_of_other_train
  (l1 : ℝ) (v1_kmph : ℝ) (v2_kmph : ℝ) (t : ℝ) : ℝ :=
  let v1 := (v1_kmph * 1000) / 3600
  let v2 := (v2_kmph * 1000) / 3600
  let relative_speed := v1 + v2
  let total_distance := relative_speed * t
  total_distance - l1

theorem other_train_length
  (l1 : ℝ) (v1_kmph : ℝ) (v2_kmph : ℝ) (t : ℝ)
  (hl1 : l1 = 230)
  (hv1 : v1_kmph = 120)
  (hv2 : v2_kmph = 80)
  (ht : t = 9) :
  length_of_other_train l1 v1_kmph v2_kmph t = 269.95 :=
by
  rw [hl1, hv1, hv2, ht]
  -- Proof steps skipped
  sorry

end NUMINAMATH_GPT_other_train_length_l373_37327


namespace NUMINAMATH_GPT_encoded_integer_one_less_l373_37397

theorem encoded_integer_one_less (BDF BEA BFB EAB : ℕ)
  (hBDF : BDF = 1 * 7^2 + 3 * 7 + 6)
  (hBEA : BEA = 1 * 7^2 + 5 * 7 + 0)
  (hBFB : BFB = 1 * 7^2 + 5 * 7 + 1)
  (hEAB : EAB = 5 * 7^2 + 0 * 7 + 1)
  : EAB - 1 = 245 :=
by
  sorry

end NUMINAMATH_GPT_encoded_integer_one_less_l373_37397


namespace NUMINAMATH_GPT_floor_x_floor_x_eq_42_l373_37354

theorem floor_x_floor_x_eq_42 (x : ℝ) : (⌊x * ⌊x⌋⌋ = 42) ↔ (7 ≤ x ∧ x < 43 / 6) :=
by sorry

end NUMINAMATH_GPT_floor_x_floor_x_eq_42_l373_37354


namespace NUMINAMATH_GPT_fraction_division_l373_37305

-- Definition of fractions involved
def frac1 : ℚ := 4 / 9
def frac2 : ℚ := 5 / 8

-- Statement of the proof problem
theorem fraction_division :
  (frac1 / frac2) = 32 / 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_division_l373_37305


namespace NUMINAMATH_GPT_root_polynomial_satisfies_expression_l373_37382

noncomputable def roots_of_polynomial (x : ℕ) : Prop :=
  x^3 - 15 * x^2 + 25 * x - 10 = 0

theorem root_polynomial_satisfies_expression (p q r : ℕ) 
    (h1 : roots_of_polynomial p)
    (h2 : roots_of_polynomial q)
    (h3 : roots_of_polynomial r)
    (h_sum : p + q + r = 15)
    (h_prod : p*q + q*r + r*p = 25) :
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 :=
by sorry

end NUMINAMATH_GPT_root_polynomial_satisfies_expression_l373_37382
