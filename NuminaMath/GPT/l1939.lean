import Mathlib

namespace NUMINAMATH_GPT_find_numerator_l1939_193911

variable {y : ℝ} (hy : y > 0) (n : ℝ)

theorem find_numerator (h: (2 * y / 10) + n = 1 / 2 * y) : n = 3 :=
sorry

end NUMINAMATH_GPT_find_numerator_l1939_193911


namespace NUMINAMATH_GPT_bowling_team_scores_l1939_193916

theorem bowling_team_scores : 
  ∀ (A B C : ℕ), 
  C = 162 → 
  B = 3 * C → 
  A + B + C = 810 → 
  A / B = 1 / 3 := 
by 
  intros A B C h1 h2 h3 
  sorry

end NUMINAMATH_GPT_bowling_team_scores_l1939_193916


namespace NUMINAMATH_GPT_unique_isolating_line_a_eq_2e_l1939_193984

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x

theorem unique_isolating_line_a_eq_2e (a : ℝ) (h : a > 0) :
  (∃ k b, ∀ x : ℝ, f x ≥ k * x + b ∧ k * x + b ≥ g a x) → a = 2 * Real.exp 1 :=
sorry

end NUMINAMATH_GPT_unique_isolating_line_a_eq_2e_l1939_193984


namespace NUMINAMATH_GPT_expression_equals_five_l1939_193998

theorem expression_equals_five (a : ℝ) (h : 2 * a^2 - 3 * a + 4 = 5) : 7 + 6 * a - 4 * a^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_five_l1939_193998


namespace NUMINAMATH_GPT_probability_all_three_blue_l1939_193931

theorem probability_all_three_blue :
  let total_jellybeans := 20
  let initial_blue := 10
  let initial_red := 10
  let prob_first_blue := initial_blue / total_jellybeans
  let prob_second_blue := (initial_blue - 1) / (total_jellybeans - 1)
  let prob_third_blue := (initial_blue - 2) / (total_jellybeans - 2)
  prob_first_blue * prob_second_blue * prob_third_blue = 2 / 19 := 
by
  sorry

end NUMINAMATH_GPT_probability_all_three_blue_l1939_193931


namespace NUMINAMATH_GPT_find_divisor_l1939_193930

theorem find_divisor (D Q R d: ℕ) (hD: D = 16698) (hQ: Q = 89) (hR: R = 14) (hDiv: D = d * Q + R): d = 187 := 
by 
  sorry

end NUMINAMATH_GPT_find_divisor_l1939_193930


namespace NUMINAMATH_GPT_integer_roots_polynomial_l1939_193993

theorem integer_roots_polynomial 
(m n : ℕ) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  (∃ a b c : ℤ, a + b + c = 17 ∧ a * b * c = n^2 ∧ a * b + b * c + c * a = m) ↔ 
  (m, n) = (80, 10) ∨ (m, n) = (88, 12) ∨ (m, n) = (80, 8) ∨ (m, n) = (90, 12) := 
sorry

end NUMINAMATH_GPT_integer_roots_polynomial_l1939_193993


namespace NUMINAMATH_GPT_find_a3_l1939_193927

noncomputable def S (n : ℕ) (a₁ q : ℚ) : ℚ :=
  a₁ * (1 - q ^ n) / (1 - q)

noncomputable def a (n : ℕ) (a₁ q : ℚ) : ℚ :=
  a₁ * q ^ (n - 1)

theorem find_a3 (a₁ q : ℚ) (h1 : S 6 a₁ q / S 3 a₁ q = -19 / 8)
  (h2 : a 4 a₁ q - a 2 a₁ q = -15 / 8) :
  a 3 a₁ q = 9 / 4 :=
by sorry

end NUMINAMATH_GPT_find_a3_l1939_193927


namespace NUMINAMATH_GPT_remainder_7_pow_253_mod_12_l1939_193917

theorem remainder_7_pow_253_mod_12 : (7 ^ 253) % 12 = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_7_pow_253_mod_12_l1939_193917


namespace NUMINAMATH_GPT_altitude_segment_length_l1939_193919

theorem altitude_segment_length 
  {A B C D E : Type} 
  (BD DC AE y : ℝ) 
  (h1 : BD = 4) 
  (h2 : DC = 6) 
  (h3 : AE = 3) 
  (h4 : 3 / 4 = 9 / (y + 3)) : 
  y = 9 := 
by 
  sorry

end NUMINAMATH_GPT_altitude_segment_length_l1939_193919


namespace NUMINAMATH_GPT_geometric_series_sum_l1939_193942

-- Definitions based on conditions
def a : ℚ := 3 / 2
def r : ℚ := -4 / 9

-- Statement of the proof
theorem geometric_series_sum : (a / (1 - r)) = 27 / 26 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1939_193942


namespace NUMINAMATH_GPT_find_difference_l1939_193948

theorem find_difference (a b : ℕ) (h1 : a < b) (h2 : a + b = 78) (h3 : Nat.lcm a b = 252) : b - a = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_difference_l1939_193948


namespace NUMINAMATH_GPT_discount_percentage_l1939_193990

theorem discount_percentage 
  (C : ℝ) (S : ℝ) (P : ℝ) (SP : ℝ)
  (h1 : C = 48)
  (h2 : 0.60 * S = C)
  (h3 : P = 16)
  (h4 : P = S - SP)
  (h5 : SP = 80 - 16)
  (h6 : S = 80) :
  (S - SP) / S * 100 = 20 := by
sorry

end NUMINAMATH_GPT_discount_percentage_l1939_193990


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1939_193943

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1939_193943


namespace NUMINAMATH_GPT_ratio_of_saramago_readers_l1939_193997

theorem ratio_of_saramago_readers 
  (W : ℕ) (S K B N : ℕ)
  (h1 : W = 42)
  (h2 : K = W / 6)
  (h3 : B = 3)
  (h4 : N = (S - B) - 1)
  (h5 : W = (S - B) + (K - B) + B + N) :
  S / W = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_saramago_readers_l1939_193997


namespace NUMINAMATH_GPT_minimum_value_y_is_2_l1939_193960

noncomputable def minimum_value_y (x : ℝ) : ℝ :=
  x + (1 / x)

theorem minimum_value_y_is_2 (x : ℝ) (hx : 0 < x) : 
  (∀ y, y = minimum_value_y x → y ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_y_is_2_l1939_193960


namespace NUMINAMATH_GPT_pounds_per_ton_l1939_193957

theorem pounds_per_ton (packet_count : ℕ) (packet_weight_pounds : ℚ) (packet_weight_ounces : ℚ) (ounces_per_pound : ℚ) (total_weight_tons : ℚ) (total_weight_pounds : ℚ) :
  packet_count = 1760 →
  packet_weight_pounds = 16 →
  packet_weight_ounces = 4 →
  ounces_per_pound = 16 →
  total_weight_tons = 13 →
  total_weight_pounds = (packet_count * (packet_weight_pounds + (packet_weight_ounces / ounces_per_pound))) →
  total_weight_pounds / total_weight_tons = 2200 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_pounds_per_ton_l1939_193957


namespace NUMINAMATH_GPT_time_per_toy_is_3_l1939_193994

-- Define the conditions
variable (total_toys : ℕ) (total_hours : ℕ)

-- Define the given condition
def given_condition := (total_toys = 50 ∧ total_hours = 150)

-- Define the statement to be proved
theorem time_per_toy_is_3 (h : given_condition total_toys total_hours) :
  total_hours / total_toys = 3 := by
sorry

end NUMINAMATH_GPT_time_per_toy_is_3_l1939_193994


namespace NUMINAMATH_GPT_product_of_two_numbers_l1939_193922

theorem product_of_two_numbers (x y : ℝ) (h_diff : x - y = 12) (h_sum_of_squares : x^2 + y^2 = 245) : x * y = 50.30 :=
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1939_193922


namespace NUMINAMATH_GPT_fraction_identity_l1939_193946

theorem fraction_identity (a b c : ℝ) (h1 : a + b + c > 0) (h2 : a + b - c > 0) (h3 : a + c - b > 0) (h4 : b + c - a > 0) 
  (h5 : (a+b+c)/(a+b-c) = 7) (h6 : (a+b+c)/(a+c-b) = 1.75) : (a+b+c)/(b+c-a) = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l1939_193946


namespace NUMINAMATH_GPT_melissa_trip_total_time_l1939_193929

theorem melissa_trip_total_time :
  ∀ (freeway_dist rural_dist : ℕ) (freeway_speed_factor : ℕ) 
  (rural_time : ℕ),
  freeway_dist = 80 →
  rural_dist = 20 →
  freeway_speed_factor = 4 →
  rural_time = 40 →
  (rural_dist * freeway_speed_factor / rural_time + freeway_dist / (rural_dist * freeway_speed_factor / rural_time)) = 80 :=
by
  intros freeway_dist rural_dist freeway_speed_factor rural_time hd1 hd2 hd3 hd4
  sorry

end NUMINAMATH_GPT_melissa_trip_total_time_l1939_193929


namespace NUMINAMATH_GPT_percentage_difference_l1939_193972

-- Define the numbers
def n : ℕ := 1600
def m : ℕ := 650

-- Define the percentages calculated
def p₁ : ℕ := (20 * n) / 100
def p₂ : ℕ := (20 * m) / 100

-- The theorem to be proved: the difference between the two percentages is 190
theorem percentage_difference : p₁ - p₂ = 190 := by
  sorry

end NUMINAMATH_GPT_percentage_difference_l1939_193972


namespace NUMINAMATH_GPT_chess_sequences_l1939_193967

def binomial (n k : Nat) : Nat := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem chess_sequences :
  binomial 11 4 = 210 := by
  sorry

end NUMINAMATH_GPT_chess_sequences_l1939_193967


namespace NUMINAMATH_GPT_min_candies_to_remove_l1939_193988

theorem min_candies_to_remove {n : ℕ} (h : n = 31) : (∃ k, (n - k) % 5 = 0) → k = 1 :=
by
  sorry

end NUMINAMATH_GPT_min_candies_to_remove_l1939_193988


namespace NUMINAMATH_GPT_carson_total_distance_l1939_193944

def perimeter (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length + width)

def total_distance (length : ℕ) (width : ℕ) (rounds : ℕ) (breaks : ℕ) (break_distance : ℕ) : ℕ :=
  let P := perimeter length width
  let distance_rounds := rounds * P
  let distance_breaks := breaks * break_distance
  distance_rounds + distance_breaks

theorem carson_total_distance :
  total_distance 600 400 8 4 100 = 16400 :=
by
  sorry

end NUMINAMATH_GPT_carson_total_distance_l1939_193944


namespace NUMINAMATH_GPT_possible_integer_roots_l1939_193937

def polynomial (x : ℤ) : ℤ := x^3 + 2 * x^2 - 3 * x - 17

theorem possible_integer_roots :
  ∃ (roots : List ℤ), roots = [1, -1, 17, -17] ∧ ∀ r ∈ roots, polynomial r = 0 := 
sorry

end NUMINAMATH_GPT_possible_integer_roots_l1939_193937


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1939_193925

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (r : ℝ)
  (h1 : a 1 + a 3 = 8)
  (h2 : a 5 + a 7 = 4)
  (geometric_seq : ∀ n, a n = a 1 * r ^ (n - 1)) :
  a 9 + a 11 + a 13 + a 15 = 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1939_193925


namespace NUMINAMATH_GPT_john_rental_weeks_l1939_193989

noncomputable def camera_value : ℝ := 5000
noncomputable def rental_fee_rate : ℝ := 0.10
noncomputable def friend_payment_rate : ℝ := 0.40
noncomputable def john_total_payment : ℝ := 1200

theorem john_rental_weeks :
  let weekly_rental_fee := camera_value * rental_fee_rate
  let friend_payment := weekly_rental_fee * friend_payment_rate
  let john_weekly_payment := weekly_rental_fee - friend_payment
  let rental_weeks := john_total_payment / john_weekly_payment
  rental_weeks = 4 :=
by
  -- Place for proof steps
  sorry

end NUMINAMATH_GPT_john_rental_weeks_l1939_193989


namespace NUMINAMATH_GPT_quadratic_greatest_value_and_real_roots_l1939_193963

theorem quadratic_greatest_value_and_real_roots :
  (∀ x : ℝ, -x^2 + 9 * x - 20 ≥ 0 → x ≤ 5)
  ∧ (∃ x : ℝ, -x^2 + 9 * x - 20 = 0)
  :=
sorry

end NUMINAMATH_GPT_quadratic_greatest_value_and_real_roots_l1939_193963


namespace NUMINAMATH_GPT_abs_sum_ge_sqrt_three_over_two_l1939_193956

open Real

theorem abs_sum_ge_sqrt_three_over_two
  (a b : ℝ) : (|a| + |b| ≥ 2 / sqrt 3) ∧ (∀ x, |a * sin x + b * sin (2 * x)| ≤ 1) ↔
  (a, b) = (4 / (3 * sqrt 3), 2 / (3 * sqrt 3)) ∨ 
  (a, b) = (-4 / (3 * sqrt 3), -2 / (3 * sqrt 3)) ∨
  (a, b) = (4 / (3 * sqrt 3), -2 / (3 * sqrt 3)) ∨
  (a, b) = (-4 / (3 * sqrt 3), 2 / (3 * sqrt 3)) := 
sorry

end NUMINAMATH_GPT_abs_sum_ge_sqrt_three_over_two_l1939_193956


namespace NUMINAMATH_GPT_contractor_engagement_days_l1939_193982

theorem contractor_engagement_days 
  (days_worked : ℕ) 
  (total_days_absent : ℕ) 
  (work_payment : ℕ → ℤ)
  (absent_fine : ℕ → ℤ)
  (total_payment : ℤ) 
  (total_days : ℕ) 
  (h1 : work_payment days_worked = 25 * days_worked)
  (h2 : absent_fine total_days_absent = 750)
  (h3 : total_payment = (work_payment days_worked) - (absent_fine total_days_absent))
  (h4 : total_payment = 425)
  (h5 : total_days_absent = 10) 
  (h6 : sorry) : -- This assumes the result of x = 20 proving work days 
  total_days = days_worked + total_days_absent := 
  by
    sorry

end NUMINAMATH_GPT_contractor_engagement_days_l1939_193982


namespace NUMINAMATH_GPT_log_base_9_of_729_l1939_193915

theorem log_base_9_of_729 : ∃ x : ℝ, (9:ℝ) = 3^2 ∧ (729:ℝ) = 3^6 ∧ (9:ℝ)^x = 729 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_log_base_9_of_729_l1939_193915


namespace NUMINAMATH_GPT_average_infect_influence_l1939_193924

theorem average_infect_influence
  (x : ℝ)
  (h : (1 + x)^2 = 100) :
  x = 9 :=
sorry

end NUMINAMATH_GPT_average_infect_influence_l1939_193924


namespace NUMINAMATH_GPT_average_speed_l1939_193983

-- Define the speeds in the first and second hours
def speed_first_hour : ℝ := 90
def speed_second_hour : ℝ := 42

-- Define the time taken for each hour
def time_first_hour : ℝ := 1
def time_second_hour : ℝ := 1

-- Calculate the total distance and total time
def total_distance : ℝ := speed_first_hour + speed_second_hour
def total_time : ℝ := time_first_hour + time_second_hour

-- State the theorem for the average speed
theorem average_speed : total_distance / total_time = 66 := by
  sorry

end NUMINAMATH_GPT_average_speed_l1939_193983


namespace NUMINAMATH_GPT_relationship_of_coefficients_l1939_193980

theorem relationship_of_coefficients (a b c : ℝ) (α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0) 
  (h_eq' : a * β^2 + b * β + c = 0) 
  (h_roots : β = 3 * α) :
  3 * b^2 = 16 * a * c := 
sorry

end NUMINAMATH_GPT_relationship_of_coefficients_l1939_193980


namespace NUMINAMATH_GPT_number_of_newborn_members_l1939_193949

theorem number_of_newborn_members (N : ℝ) (h : (9/10 : ℝ) ^ 3 * N = 291.6) : N = 400 :=
sorry

end NUMINAMATH_GPT_number_of_newborn_members_l1939_193949


namespace NUMINAMATH_GPT_wendy_albums_l1939_193901

theorem wendy_albums (total_pictures remaining_pictures pictures_per_album : ℕ) 
    (h1 : total_pictures = 79)
    (h2 : remaining_pictures = total_pictures - 44)
    (h3 : pictures_per_album = 7) :
    remaining_pictures / pictures_per_album = 5 := by
  sorry

end NUMINAMATH_GPT_wendy_albums_l1939_193901


namespace NUMINAMATH_GPT_Tom_final_balance_l1939_193996

theorem Tom_final_balance :
  let initial_allowance := 12
  let week1_spending := initial_allowance / 3
  let balance_after_week1 := initial_allowance - week1_spending
  let week2_spending := balance_after_week1 / 4
  let balance_after_week2 := balance_after_week1 - week2_spending
  let additional_earning := 5
  let balance_after_earning := balance_after_week2 + additional_earning
  let week3_spending := balance_after_earning / 2
  let balance_after_week3 := balance_after_earning - week3_spending
  let penultimate_day_spending := 3
  let final_balance := balance_after_week3 - penultimate_day_spending
  final_balance = 2.50 :=
by
  sorry

end NUMINAMATH_GPT_Tom_final_balance_l1939_193996


namespace NUMINAMATH_GPT_inequality_proof_l1939_193987

theorem inequality_proof (x y : ℝ) : 
  -1 / 2 ≤ (x + y) * (1 - x * y) / ((1 + x ^ 2) * (1 + y ^ 2)) ∧
  (x + y) * (1 - x * y) / ((1 + x ^ 2) * (1 + y ^ 2)) ≤ 1 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1939_193987


namespace NUMINAMATH_GPT_quadratic_root_a_l1939_193973

theorem quadratic_root_a {a : ℝ} (h : (2 : ℝ) ∈ {x : ℝ | x^2 + 3 * x + a = 0}) : a = -10 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_a_l1939_193973


namespace NUMINAMATH_GPT_sachin_age_l1939_193978

variable {S R : ℕ}

theorem sachin_age
  (h1 : R = S + 7)
  (h2 : S * 3 = 2 * R) :
  S = 14 :=
sorry

end NUMINAMATH_GPT_sachin_age_l1939_193978


namespace NUMINAMATH_GPT_focus_on_negative_y_axis_l1939_193938

-- Definition of the condition: equation of the parabola
def parabola (x y : ℝ) := x^2 + y = 0

-- Statement of the problem
theorem focus_on_negative_y_axis (x y : ℝ) (h : parabola x y) : 
  -- The focus of the parabola lies on the negative half of the y-axis
  ∃ y, y < 0 :=
sorry

end NUMINAMATH_GPT_focus_on_negative_y_axis_l1939_193938


namespace NUMINAMATH_GPT_solution_of_inequality_system_l1939_193936

theorem solution_of_inequality_system (a b : ℝ) 
    (h1 : 4 - 2 * a = 0)
    (h2 : (3 + b) / 2 = 1) : a + b = 1 := 
by 
  sorry

end NUMINAMATH_GPT_solution_of_inequality_system_l1939_193936


namespace NUMINAMATH_GPT_john_annual_profit_l1939_193986

-- Definitions of monthly incomes
def TenantA_income : ℕ := 350
def TenantB_income : ℕ := 400
def TenantC_income : ℕ := 450

-- Total monthly income
def total_monthly_income : ℕ := TenantA_income + TenantB_income + TenantC_income

-- Definitions of monthly expenses
def rent_expense : ℕ := 900
def utilities_expense : ℕ := 100
def maintenance_fee : ℕ := 50

-- Total monthly expenses
def total_monthly_expense : ℕ := rent_expense + utilities_expense + maintenance_fee

-- Monthly profit
def monthly_profit : ℕ := total_monthly_income - total_monthly_expense

-- Annual profit
def annual_profit : ℕ := monthly_profit * 12

theorem john_annual_profit :
  annual_profit = 1800 := by
  -- The proof is omitted, but the statement asserts that John makes an annual profit of $1800.
  sorry

end NUMINAMATH_GPT_john_annual_profit_l1939_193986


namespace NUMINAMATH_GPT_sin_double_angle_l1939_193921

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1939_193921


namespace NUMINAMATH_GPT_cube_volume_multiple_of_6_l1939_193932

theorem cube_volume_multiple_of_6 (n : ℕ) (h : ∃ m : ℕ, n^3 = 24 * m) : ∃ k : ℕ, n = 6 * k :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_multiple_of_6_l1939_193932


namespace NUMINAMATH_GPT_fraction_given_to_friend_l1939_193934

theorem fraction_given_to_friend (s u r g k : ℕ) 
  (h1: s = 135) 
  (h2: u = s / 3) 
  (h3: r = s - u) 
  (h4: k = 54) 
  (h5: g = r - k) :
  g / r = 2 / 5 := 
  by
  sorry

end NUMINAMATH_GPT_fraction_given_to_friend_l1939_193934


namespace NUMINAMATH_GPT_problem_l1939_193904

theorem problem (a : ℕ) (h1 : a = 444) : (444 ^ 444) % 13 = 1 :=
by
  have h444 : 444 % 13 = 3 := by sorry
  have h3_pow3 : 3 ^ 3 % 13 = 1 := by sorry
  sorry

end NUMINAMATH_GPT_problem_l1939_193904


namespace NUMINAMATH_GPT_infinite_solutions_exists_l1939_193953

theorem infinite_solutions_exists :
  ∃ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x ≠ y) ∧ (y ≠ z) ∧ (z ≠ x) ∧
  (x - y + z = 1) ∧ ((x * y) % z = 0) ∧ ((y * z) % x = 0) ∧ ((z * x) % y = 0) ∧
  ∀ n : ℕ, ∃ x y z : ℕ, (n > 0) ∧ (x = n * (n^2 + n - 1)) ∧ (y = (n+1) * (n^2 + n - 1)) ∧ (z = n * (n+1)) := by
  sorry

end NUMINAMATH_GPT_infinite_solutions_exists_l1939_193953


namespace NUMINAMATH_GPT_range_of_a_l1939_193969

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = a * Real.log x + 1/2 * x^2)
  (h_ineq : ∀ x1 x2 : ℝ, x1 ≠ x2 → 0 < x1 → 0 < x2 → (f x1 - f x2) / (x1 - x2) > 4) : a > 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1939_193969


namespace NUMINAMATH_GPT_broker_wealth_increase_after_two_years_l1939_193900

theorem broker_wealth_increase_after_two_years :
  let initial_investment : ℝ := 100
  let first_year_increase : ℝ := 0.75
  let second_year_decrease : ℝ := 0.30
  let end_first_year := initial_investment * (1 + first_year_increase)
  let end_second_year := end_first_year * (1 - second_year_decrease)
  end_second_year - initial_investment = 22.50 :=
by
  sorry

end NUMINAMATH_GPT_broker_wealth_increase_after_two_years_l1939_193900


namespace NUMINAMATH_GPT_sqrt_x_plus_5_l1939_193966

theorem sqrt_x_plus_5 (x : ℝ) (h : x = -1) : Real.sqrt (x + 5) = 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_x_plus_5_l1939_193966


namespace NUMINAMATH_GPT_claire_photos_l1939_193945

-- Define the number of photos taken by Claire, Lisa, and Robert
variables (C L R : ℕ)

-- Conditions based on the problem
def Lisa_photos (C : ℕ) := 3 * C
def Robert_photos (C : ℕ) := C + 24

-- Prove that C = 12 given the conditions
theorem claire_photos : 
  (L = Lisa_photos C) ∧ (R = Robert_photos C) ∧ (L = R) → C = 12 := 
by
  sorry

end NUMINAMATH_GPT_claire_photos_l1939_193945


namespace NUMINAMATH_GPT_find_x_l1939_193905

theorem find_x (x : ℝ) (h : 121 * x^4 = 75625) : x = 5 :=
sorry

end NUMINAMATH_GPT_find_x_l1939_193905


namespace NUMINAMATH_GPT_find_slope_of_chord_l1939_193970

noncomputable def slope_of_chord (x1 x2 y1 y2 : ℝ) : ℝ :=
  (y1 - y2) / (x1 - x2)

theorem find_slope_of_chord :
  (∀ (x y : ℝ), x^2 / 36 + y^2 / 9 = 1 → ∃ (x1 x2 y1 y2 : ℝ),
    x1 + x2 = 8 ∧ y1 + y2 = 4 ∧ x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2 ∧ slope_of_chord x1 x2 y1 y2 = -1 / 2) := sorry

end NUMINAMATH_GPT_find_slope_of_chord_l1939_193970


namespace NUMINAMATH_GPT_egyptian_fraction_l1939_193999

theorem egyptian_fraction (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) : 
  (2 : ℚ) / 7 = (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c :=
by
  sorry

end NUMINAMATH_GPT_egyptian_fraction_l1939_193999


namespace NUMINAMATH_GPT_vacation_total_cost_l1939_193941

def plane_ticket_cost (per_person_cost : ℕ) (num_people : ℕ) : ℕ :=
  num_people * per_person_cost

def hotel_stay_cost (per_person_per_day_cost : ℕ) (num_people : ℕ) (num_days : ℕ) : ℕ :=
  num_people * per_person_per_day_cost * num_days

def total_vacation_cost (plane_ticket_cost : ℕ) (hotel_stay_cost : ℕ) : ℕ :=
  plane_ticket_cost + hotel_stay_cost

theorem vacation_total_cost :
  let per_person_plane_ticket_cost := 24
  let per_person_hotel_cost := 12
  let num_people := 2
  let num_days := 3
  let plane_cost := plane_ticket_cost per_person_plane_ticket_cost num_people
  let hotel_cost := hotel_stay_cost per_person_hotel_cost num_people num_days
  total_vacation_cost plane_cost hotel_cost = 120 := by
  sorry

end NUMINAMATH_GPT_vacation_total_cost_l1939_193941


namespace NUMINAMATH_GPT_total_children_l1939_193940

-- Given the conditions
def toy_cars : Nat := 134
def dolls : Nat := 269

-- Prove that the total number of children is 403
theorem total_children (h_cars : toy_cars = 134) (h_dolls : dolls = 269) :
  toy_cars + dolls = 403 :=
by
  sorry

end NUMINAMATH_GPT_total_children_l1939_193940


namespace NUMINAMATH_GPT_prism_dimensions_l1939_193991

theorem prism_dimensions (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 45) (h3 : b * c = 60) : 
  a = 7.2 ∧ b = 9.6 ∧ c = 14.4 :=
by {
  -- Proof skipped for now
  sorry
}

end NUMINAMATH_GPT_prism_dimensions_l1939_193991


namespace NUMINAMATH_GPT_prob_at_least_3_correct_l1939_193977

-- Define the probability of one patient being cured
def prob_cured : ℝ := 0.9

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the probability of exactly 3 out of 4 patients being cured
def prob_exactly_3 : ℝ :=
  binomial 4 3 * prob_cured^3 * (1 - prob_cured)

-- Define the probability of all 4 patients being cured
def prob_all_4 : ℝ :=
  prob_cured^4

-- Define the probability of at least 3 out of 4 patients being cured
def prob_at_least_3 : ℝ :=
  prob_exactly_3 + prob_all_4

-- The theorem to prove
theorem prob_at_least_3_correct : prob_at_least_3 = 0.9477 :=
  by
  sorry

end NUMINAMATH_GPT_prob_at_least_3_correct_l1939_193977


namespace NUMINAMATH_GPT_badminton_tournament_l1939_193962

theorem badminton_tournament (n x : ℕ) (h1 : 2 * n > 0) (h2 : 3 * n > 0) (h3 : (5 * n) * (5 * n - 1) = 14 * x) : n = 3 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_badminton_tournament_l1939_193962


namespace NUMINAMATH_GPT_work_days_A_l1939_193912

theorem work_days_A (x : ℝ) (h1 : ∀ y : ℝ, y = 20) (h2 : ∀ z : ℝ, z = 5) 
  (h3 : ∀ w : ℝ, w = 0.41666666666666663) :
  x = 15 :=
  sorry

end NUMINAMATH_GPT_work_days_A_l1939_193912


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1939_193913

-- Defining the geometric sequence related properties and conditions
theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * r) → 
  S 3 = a 0 + a 1 + a 2 →
  S 6 = a 3 + a 4 + a 5 →
  S 12 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 →
  S 3 = 3 →
  S 6 = 6 →
  S 12 = 45 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1939_193913


namespace NUMINAMATH_GPT_variance_transformation_l1939_193902

theorem variance_transformation (a1 a2 a3 : ℝ) 
  (h1 : (a1 + a2 + a3) / 3 = 4) 
  (h2 : ((a1 - 4)^2 + (a2 - 4)^2 + (a3 - 4)^2) / 3 = 3) : 
  ((3 * a1 - 2 - (3 * 4 - 2))^2 + (3 * a2 - 2 - (3 * 4 - 2))^2 + (3 * a3 - 2 - (3 * 4 - 2))^2) / 3 = 27 := 
sorry

end NUMINAMATH_GPT_variance_transformation_l1939_193902


namespace NUMINAMATH_GPT_buffaloes_added_l1939_193958

-- Let B be the daily fodder consumption of one buffalo in units
noncomputable def daily_fodder_buffalo (B : ℝ) := B
noncomputable def daily_fodder_cow (B : ℝ) := (3 / 4) * B
noncomputable def daily_fodder_ox (B : ℝ) := (3 / 2) * B

-- Initial conditions
def initial_buffaloes := 15
def initial_cows := 24
def initial_oxen := 8
def initial_days := 24
noncomputable def total_initial_fodder (B : ℝ) := (initial_buffaloes * daily_fodder_buffalo B) + (initial_oxen * daily_fodder_ox B) + (initial_cows * daily_fodder_cow B)
noncomputable def total_fodder (B : ℝ) := total_initial_fodder B * initial_days

-- New conditions after adding cows and buffaloes
def additional_cows := 60
def new_days := 9
noncomputable def total_new_daily_fodder (B : ℝ) (x : ℝ) := ((initial_buffaloes + x) * daily_fodder_buffalo B) + (initial_oxen * daily_fodder_ox B) + ((initial_cows + additional_cows) * daily_fodder_cow B)

-- Proof statement: Prove that given the conditions, the number of additional buffaloes, x, is 30.
theorem buffaloes_added (B : ℝ) : 
  (total_fodder B = total_new_daily_fodder B 30 * new_days) :=
by sorry

end NUMINAMATH_GPT_buffaloes_added_l1939_193958


namespace NUMINAMATH_GPT_cubic_polynomial_roots_l1939_193985

noncomputable def polynomial := fun x : ℝ => x^3 - 2*x - 2

theorem cubic_polynomial_roots
  (x y z : ℝ) 
  (h1: polynomial x = 0)
  (h2: polynomial y = 0)
  (h3: polynomial z = 0):
  x * (y - z)^2 + y * (z - x)^2 + z * (x - y)^2 = 0 :=
by
  -- Solution steps will be filled here to prove the theorem
  sorry

end NUMINAMATH_GPT_cubic_polynomial_roots_l1939_193985


namespace NUMINAMATH_GPT_determine_x_l1939_193995

theorem determine_x (A B C : ℝ) (x : ℝ) (h1 : C > B) (h2 : B > A) (h3 : A > 0)
  (h4 : A = B - (x / 100) * B) (h5 : C = A + 2 * B) :
  x = 100 * ((B - A) / B) :=
sorry

end NUMINAMATH_GPT_determine_x_l1939_193995


namespace NUMINAMATH_GPT_quotient_A_div_B_l1939_193906

-- Define A according to the given conditions
def A : ℕ := (8 * 10) + (13 * 1)

-- Define B according to the given conditions
def B : ℕ := 30 - 9 - 9 - 9

-- Prove that the quotient of A divided by B is 31
theorem quotient_A_div_B : (A / B) = 31 := by
  sorry

end NUMINAMATH_GPT_quotient_A_div_B_l1939_193906


namespace NUMINAMATH_GPT_sum_of_areas_squares_l1939_193981

theorem sum_of_areas_squares (a : ℕ) (h1 : (a + 4)^2 - a^2 = 80) : a^2 + (a + 4)^2 = 208 := by
  sorry

end NUMINAMATH_GPT_sum_of_areas_squares_l1939_193981


namespace NUMINAMATH_GPT_girls_in_school_play_l1939_193939

theorem girls_in_school_play (G : ℕ) (boys : ℕ) (total_parents : ℕ)
  (h1 : boys = 8) (h2 : total_parents = 28) (h3 : 2 * boys + 2 * G = total_parents) : 
  G = 6 :=
sorry

end NUMINAMATH_GPT_girls_in_school_play_l1939_193939


namespace NUMINAMATH_GPT_runner_time_difference_l1939_193971

theorem runner_time_difference (v : ℝ) (h1 : 0 < v) (h2 : 0 < 20 / v) (h3 : 8 = 40 / v) :
  8 - (20 / v) = 4 := by
  sorry

end NUMINAMATH_GPT_runner_time_difference_l1939_193971


namespace NUMINAMATH_GPT_jerry_age_l1939_193965

theorem jerry_age (M J : ℝ) (h₁ : M = 17) (h₂ : M = 2.5 * J - 3) : J = 8 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_jerry_age_l1939_193965


namespace NUMINAMATH_GPT_heloise_gives_dogs_to_janet_l1939_193964

theorem heloise_gives_dogs_to_janet :
  ∃ d c : ℕ, d * 17 = c * 10 ∧ d + c = 189 ∧ d - 60 = 10 :=
by
  sorry

end NUMINAMATH_GPT_heloise_gives_dogs_to_janet_l1939_193964


namespace NUMINAMATH_GPT_power_ineq_for_n_geq_5_l1939_193920

noncomputable def power_ineq (n : ℕ) : Prop := 2^n > n^2 + 1

theorem power_ineq_for_n_geq_5 (n : ℕ) (h : n ≥ 5) : power_ineq n :=
  sorry

end NUMINAMATH_GPT_power_ineq_for_n_geq_5_l1939_193920


namespace NUMINAMATH_GPT_cubes_with_4_neighbors_l1939_193951

theorem cubes_with_4_neighbors (a b c : ℕ) (h₁ : 3 < a) (h₂ : 3 < b) (h₃ : 3 < c)
  (h₄ : (a - 2) * (b - 2) * (c - 2) = 429) : 
  4 * ((a - 2) + (b - 2) + (c - 2)) = 108 := by
  sorry

end NUMINAMATH_GPT_cubes_with_4_neighbors_l1939_193951


namespace NUMINAMATH_GPT_compare_sqrts_l1939_193979

theorem compare_sqrts (a b c : ℝ) (h1 : a = 2 * Real.sqrt 7) (h2 : b = 3 * Real.sqrt 5) (h3 : c = 5 * Real.sqrt 2):
  c > b ∧ b > a :=
by
  sorry

end NUMINAMATH_GPT_compare_sqrts_l1939_193979


namespace NUMINAMATH_GPT_fraction_of_second_year_given_not_third_year_l1939_193909

theorem fraction_of_second_year_given_not_third_year (total_students : ℕ) 
  (third_year_students : ℕ) (second_year_students : ℕ) :
  third_year_students = total_students * 30 / 100 →
  second_year_students = total_students * 10 / 100 →
  ↑second_year_students / (total_students - third_year_students) = (1 : ℚ) / 7 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_fraction_of_second_year_given_not_third_year_l1939_193909


namespace NUMINAMATH_GPT_cos_double_angle_l1939_193947

theorem cos_double_angle (α : ℝ) (h : Real.sin (π / 2 - α) = 1 / 4) : 
  Real.cos (2 * α) = -7 / 8 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l1939_193947


namespace NUMINAMATH_GPT_minimum_value_xy_l1939_193933

theorem minimum_value_xy (x y : ℝ) (h : (x + Real.sqrt (x^2 + 1)) * (y + Real.sqrt (y^2 + 1)) ≥ 1) : x + y ≥ 0 :=
sorry

end NUMINAMATH_GPT_minimum_value_xy_l1939_193933


namespace NUMINAMATH_GPT_solve_for_n_l1939_193974

theorem solve_for_n (n : ℝ) (h : 0.05 * n + 0.1 * (30 + n) - 0.02 * n = 15.5) : n = 96 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_n_l1939_193974


namespace NUMINAMATH_GPT_positive_diff_solutions_l1939_193961

theorem positive_diff_solutions (x1 x2 : ℝ) (h1 : 2 * x1 - 3 = 14) (h2 : 2 * x2 - 3 = -14) : 
  x1 - x2 = 14 := 
by
  sorry

end NUMINAMATH_GPT_positive_diff_solutions_l1939_193961


namespace NUMINAMATH_GPT_alpha_sufficient_not_necessary_l1939_193903

def A := {x : ℝ | 2 < x ∧ x < 3}

def B (α : ℝ) := {x : ℝ | (x + 2) * (x - α) < 0}

theorem alpha_sufficient_not_necessary (α : ℝ) : 
  (α = 1 → A ∩ B α = ∅) ∧ (∃ β : ℝ, β ≠ 1 ∧ A ∩ B β = ∅) :=
by
  sorry

end NUMINAMATH_GPT_alpha_sufficient_not_necessary_l1939_193903


namespace NUMINAMATH_GPT_find_f_at_2_l1939_193959

def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x - 8

theorem find_f_at_2 (a b c : ℝ) (h : f (-2) a b c = 10) : f 2 a b c = -26 :=
by
  sorry

end NUMINAMATH_GPT_find_f_at_2_l1939_193959


namespace NUMINAMATH_GPT_find_inverse_l1939_193910

noncomputable def inverse_matrix_2x2 (a b c d : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  if ad_bc : (a * d - b * c) = 0 then (0, 0, 0, 0)
  else (d / (a * d - b * c), -b / (a * d - b * c), -c / (a * d - b * c), a / (a * d - b * c))

theorem find_inverse :
  inverse_matrix_2x2 5 7 2 3 = (3, -7, -2, 5) :=
by 
  sorry

end NUMINAMATH_GPT_find_inverse_l1939_193910


namespace NUMINAMATH_GPT_sin_315_degree_l1939_193918

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_315_degree_l1939_193918


namespace NUMINAMATH_GPT_selling_price_of_mixture_per_litre_l1939_193976

def cost_per_litre : ℝ := 3.60
def litres_of_pure_milk : ℝ := 25
def litres_of_water : ℝ := 5
def total_volume_of_mixture : ℝ := litres_of_pure_milk + litres_of_water
def total_cost_of_pure_milk : ℝ := cost_per_litre * litres_of_pure_milk

theorem selling_price_of_mixture_per_litre :
  total_cost_of_pure_milk / total_volume_of_mixture = 3 := by
  sorry

end NUMINAMATH_GPT_selling_price_of_mixture_per_litre_l1939_193976


namespace NUMINAMATH_GPT_find_symmetric_curve_equation_l1939_193992

def equation_of_curve_symmetric_to_line : Prop :=
  ∀ (x y : ℝ), (5 * x^2 + 12 * x * y - 22 * x - 12 * y - 19 = 0 ∧ x - y + 2 = 0) →
  12 * x * y + 5 * y^2 - 78 * y + 45 = 0

theorem find_symmetric_curve_equation : equation_of_curve_symmetric_to_line :=
sorry

end NUMINAMATH_GPT_find_symmetric_curve_equation_l1939_193992


namespace NUMINAMATH_GPT_inequality_with_conditions_l1939_193950

variable {a b c : ℝ}

theorem inequality_with_conditions (h : a * b + b * c + c * a = 1) :
  (|a - b| / |1 + c^2|) + (|b - c| / |1 + a^2|) ≥ (|c - a| / |1 + b^2|) :=
by
  sorry

end NUMINAMATH_GPT_inequality_with_conditions_l1939_193950


namespace NUMINAMATH_GPT_ratio_of_a_to_c_l1939_193928

variable {a b c d : ℚ}

theorem ratio_of_a_to_c (h₁ : a / b = 5 / 4) (h₂ : c / d = 4 / 3) (h₃ : d / b = 1 / 5) : 
  a / c = 75 / 16 := 
sorry

end NUMINAMATH_GPT_ratio_of_a_to_c_l1939_193928


namespace NUMINAMATH_GPT_parametric_to_general_eq_l1939_193926

theorem parametric_to_general_eq (x y θ : ℝ) 
  (h1 : x = 2 + Real.sin θ ^ 2) 
  (h2 : y = -1 + Real.cos (2 * θ)) : 
  2 * x + y - 4 = 0 ∧ 2 ≤ x ∧ x ≤ 3 := 
sorry

end NUMINAMATH_GPT_parametric_to_general_eq_l1939_193926


namespace NUMINAMATH_GPT_circumscribed_circle_radius_l1939_193908

theorem circumscribed_circle_radius (b c : ℝ) (cosA : ℝ)
  (hb : b = 2) (hc : c = 3) (hcosA : cosA = 1 / 3) : 
  R = 9 * Real.sqrt 2 / 8 :=
by
  sorry

end NUMINAMATH_GPT_circumscribed_circle_radius_l1939_193908


namespace NUMINAMATH_GPT_jack_weight_52_l1939_193975

theorem jack_weight_52 (Sam Jack : ℕ) (h1 : Sam + Jack = 96) (h2 : Jack = Sam + 8) : Jack = 52 := 
by
  sorry

end NUMINAMATH_GPT_jack_weight_52_l1939_193975


namespace NUMINAMATH_GPT_calc_x_squared_plus_5xy_plus_y_squared_l1939_193923

theorem calc_x_squared_plus_5xy_plus_y_squared 
  (x y : ℝ) 
  (h1 : x * y = 4)
  (h2 : x - y = 5) :
  x^2 + 5 * x * y + y^2 = 53 :=
by 
  sorry

end NUMINAMATH_GPT_calc_x_squared_plus_5xy_plus_y_squared_l1939_193923


namespace NUMINAMATH_GPT_train_crossing_time_l1939_193952

noncomputable def train_length : ℝ := 385
noncomputable def train_speed_kmph : ℝ := 90
noncomputable def bridge_length : ℝ := 1250

noncomputable def convert_speed_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def time_to_cross_bridge (train_length bridge_length train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_mps := convert_speed_to_mps train_speed_kmph
  total_distance / speed_mps

theorem train_crossing_time :
  time_to_cross_bridge train_length bridge_length train_speed_kmph = 65.4 :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1939_193952


namespace NUMINAMATH_GPT_problem_ab_plus_a_plus_b_l1939_193907

noncomputable def polynomial := fun x : ℝ => x^4 - 6 * x - 2

theorem problem_ab_plus_a_plus_b :
  ∀ (a b : ℝ), polynomial a = 0 → polynomial b = 0 → (a * b + a + b) = 4 :=
by
  intros a b ha hb
  sorry

end NUMINAMATH_GPT_problem_ab_plus_a_plus_b_l1939_193907


namespace NUMINAMATH_GPT_beam_reflection_equation_l1939_193954

theorem beam_reflection_equation:
  ∃ (line : ℝ → ℝ → Prop), 
  (∀ (x y : ℝ), line x y ↔ (5 * x - 2 * y - 10 = 0)) ∧
  (line 4 5) ∧ 
  (line 2 0) :=
by
  sorry

end NUMINAMATH_GPT_beam_reflection_equation_l1939_193954


namespace NUMINAMATH_GPT_direction_cosines_l1939_193914

theorem direction_cosines (x y z : ℝ) (α β γ : ℝ)
  (h1 : 2 * x - 3 * y - 3 * z - 9 = 0)
  (h2 : x - 2 * y + z + 3 = 0) :
  α = 9 / Real.sqrt 107 ∧ β = 5 / Real.sqrt 107 ∧ γ = 1 / Real.sqrt 107 :=
by
  -- Here, we will sketch out the proof to establish that these values for α, β, and γ hold.
  sorry

end NUMINAMATH_GPT_direction_cosines_l1939_193914


namespace NUMINAMATH_GPT_spherical_coords_standard_form_l1939_193935

theorem spherical_coords_standard_form :
  ∀ (ρ θ φ : ℝ), ρ > 0 → 0 ≤ θ ∧ θ < 2 * Real.pi → 0 ≤ φ ∧ φ ≤ Real.pi →
  (5, (5 * Real.pi) / 7, (11 * Real.pi) / 6) = (ρ, θ, φ) →
  (ρ, (12 * Real.pi) / 7, Real.pi / 6) = (ρ, θ, φ) :=
by 
  intros ρ θ φ hρ hθ hφ h_eq
  sorry

end NUMINAMATH_GPT_spherical_coords_standard_form_l1939_193935


namespace NUMINAMATH_GPT_intercepts_of_line_l1939_193968

theorem intercepts_of_line (x y : ℝ) 
  (h : 2 * x + 7 * y = 35) :
  (y = 5 → x = 0) ∧ (x = 17.5 → y = 0)  :=
by
  sorry

end NUMINAMATH_GPT_intercepts_of_line_l1939_193968


namespace NUMINAMATH_GPT_sixteen_grams_on_left_pan_l1939_193955

theorem sixteen_grams_on_left_pan :
  ∃ (weights : ℕ → ℕ) (pans : ℕ → ℕ) (n : ℕ),
    weights n = 16 ∧
    pans 0 = 11111 ∧
    ∃ k, (∀ i < k, weights i = 2 ^ i) ∧
    (∀ i < k, (pans 1 + weights i = 38) ∧ (pans 0 + 11111 = weights i + skeletal)) ∧
    k = 6 := by
  sorry

end NUMINAMATH_GPT_sixteen_grams_on_left_pan_l1939_193955
