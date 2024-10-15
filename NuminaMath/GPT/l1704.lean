import Mathlib

namespace NUMINAMATH_GPT_linear_correlation_l1704_170469

variable (r : ℝ) (r_critical : ℝ)

theorem linear_correlation (h1 : r = -0.9362) (h2 : r_critical = 0.8013) :
  |r| > r_critical :=
by
  sorry

end NUMINAMATH_GPT_linear_correlation_l1704_170469


namespace NUMINAMATH_GPT_local_minimum_at_1_1_l1704_170406

noncomputable def function (x y : ℝ) : ℝ :=
  x^3 + y^3 - 3 * x * y

theorem local_minimum_at_1_1 : 
  ∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ (∀ (z : ℝ), z = function x y → z = -1) :=
sorry

end NUMINAMATH_GPT_local_minimum_at_1_1_l1704_170406


namespace NUMINAMATH_GPT_inverse_function_b_value_l1704_170403

theorem inverse_function_b_value (b : ℝ) :
  (∀ x, ∃ y, 2^x + b = y) ∧ (∃ x, ∃ y, (x, y) = (2, 5)) → b = 1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_b_value_l1704_170403


namespace NUMINAMATH_GPT_find_K_values_l1704_170439

theorem find_K_values (K M : ℕ) (h1 : (K * (K + 1)) / 2 = M^2) (h2 : M < 200) (h3 : K > M) :
  K = 8 ∨ K = 49 :=
sorry

end NUMINAMATH_GPT_find_K_values_l1704_170439


namespace NUMINAMATH_GPT_position_of_2017_in_arithmetic_sequence_l1704_170455

theorem position_of_2017_in_arithmetic_sequence :
  ∀ (n : ℕ), 4 + 3 * (n - 1) = 2017 → n = 672 :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_position_of_2017_in_arithmetic_sequence_l1704_170455


namespace NUMINAMATH_GPT_find_t_l1704_170412

theorem find_t (c o u n t s : ℕ)
    (hc : c ≠ 0) (ho : o ≠ 0) (hn : n ≠ 0) (ht : t ≠ 0) (hs : s ≠ 0)
    (h1 : c + o = u)
    (h2 : u + n = t + 1)
    (h3 : t + c = s)
    (h4 : o + n + s = 15) :
    t = 7 := 
sorry

end NUMINAMATH_GPT_find_t_l1704_170412


namespace NUMINAMATH_GPT_amoeba_population_at_11am_l1704_170495

/-- Sarah observes an amoeba colony where initially there are 50 amoebas at 10:00 a.m. The population triples every 10 minutes and there are no deaths among the amoebas. Prove that the number of amoebas at 11:00 a.m. is 36450. -/
theorem amoeba_population_at_11am : 
  let initial_population := 50
  let growth_rate := 3
  let increments := 6  -- since 60 minutes / 10 minutes per increment = 6
  initial_population * (growth_rate ^ increments) = 36450 :=
by
  sorry

end NUMINAMATH_GPT_amoeba_population_at_11am_l1704_170495


namespace NUMINAMATH_GPT_range_of_a_l1704_170477

theorem range_of_a (a : ℝ) (h : ¬ ∃ x : ℝ, x^2 + (a + 1) * x + 1 ≤ 0) : -3 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1704_170477


namespace NUMINAMATH_GPT_max_area_of_equilateral_triangle_in_rectangle_l1704_170462

noncomputable def maxEquilateralTriangleArea (a b : ℝ) : ℝ :=
  if h : a ≤ b then
    (a^2 * Real.sqrt 3) / 4
  else
    (b^2 * Real.sqrt 3) / 4

theorem max_area_of_equilateral_triangle_in_rectangle :
  maxEquilateralTriangleArea 12 14 = 36 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_equilateral_triangle_in_rectangle_l1704_170462


namespace NUMINAMATH_GPT_josh_initial_wallet_l1704_170478

noncomputable def initial_wallet_amount (investment final_wallet: ℕ) (stock_increase_percentage: ℕ): ℕ :=
  let investment_value_after_rise := investment + (investment * stock_increase_percentage / 100)
  final_wallet - investment_value_after_rise

theorem josh_initial_wallet : initial_wallet_amount 2000 2900 30 = 300 :=
by
  sorry

end NUMINAMATH_GPT_josh_initial_wallet_l1704_170478


namespace NUMINAMATH_GPT_problem_l1704_170402

theorem problem (x y z : ℕ) (hx : x < 9) (hy : y < 9) (hz : z < 9) 
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z % 9 = 0) :=
sorry

end NUMINAMATH_GPT_problem_l1704_170402


namespace NUMINAMATH_GPT_probability_factor_120_less_9_l1704_170427

theorem probability_factor_120_less_9 : 
  ∀ n : ℕ, n = 120 → (∃ p : ℚ, p = 7 / 16 ∧ (∃ factors_less_9 : ℕ, factors_less_9 < 16 ∧ factors_less_9 = 7)) := 
by 
  sorry

end NUMINAMATH_GPT_probability_factor_120_less_9_l1704_170427


namespace NUMINAMATH_GPT_some_number_value_l1704_170426

theorem some_number_value (a : ℕ) (x : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * x * 49) : x = 9 := by
  sorry

end NUMINAMATH_GPT_some_number_value_l1704_170426


namespace NUMINAMATH_GPT_micheal_work_separately_40_days_l1704_170405

-- Definitions based on the problem conditions
def work_complete_together (M A : ℕ) : Prop := (1/(M:ℝ) + 1/(A:ℝ) = 1/20)
def remaining_work_completed_by_adam (A : ℕ) : Prop := (1/(A:ℝ) = 1/40)

-- The theorem we want to prove
theorem micheal_work_separately_40_days (M A : ℕ) 
  (h1 : work_complete_together M A) 
  (h2 : remaining_work_completed_by_adam A) : 
  M = 40 := 
by 
  sorry  -- Placeholder for proof

end NUMINAMATH_GPT_micheal_work_separately_40_days_l1704_170405


namespace NUMINAMATH_GPT_totalCost_l1704_170423
-- Importing the necessary library

-- Defining the conditions
def numberOfHotDogs : Nat := 6
def costPerHotDog : Nat := 50

-- Proving the total cost
theorem totalCost : numberOfHotDogs * costPerHotDog = 300 := by
  sorry

end NUMINAMATH_GPT_totalCost_l1704_170423


namespace NUMINAMATH_GPT_total_distance_of_bus_rides_l1704_170476

theorem total_distance_of_bus_rides :
  let vince_distance   := 5 / 8
  let zachary_distance := 1 / 2
  let alice_distance   := 17 / 20
  let rebecca_distance := 2 / 5
  let total_distance   := vince_distance + zachary_distance + alice_distance + rebecca_distance
  total_distance = 19/8 := by
  sorry

end NUMINAMATH_GPT_total_distance_of_bus_rides_l1704_170476


namespace NUMINAMATH_GPT_triangle_side_ineq_l1704_170466

theorem triangle_side_ineq (a b c : ℝ) 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (a - b) / (a + b) + (b - c) / (b + c) + (c - a) / (a + c) < 1 / 16 :=
  sorry

end NUMINAMATH_GPT_triangle_side_ineq_l1704_170466


namespace NUMINAMATH_GPT_Lindsay_has_26_more_black_brown_dolls_than_blonde_l1704_170449

def blonde_dolls : Nat := 4
def brown_dolls : Nat := 4 * blonde_dolls
def black_dolls : Nat := brown_dolls - 2
def total_black_brown_dolls : Nat := black_dolls + brown_dolls
def extra_black_brown_dolls (blonde_dolls black_dolls brown_dolls : Nat) : Nat :=
  total_black_brown_dolls - blonde_dolls

theorem Lindsay_has_26_more_black_brown_dolls_than_blonde :
  extra_black_brown_dolls blonde_dolls black_dolls brown_dolls = 26 := by
  sorry

end NUMINAMATH_GPT_Lindsay_has_26_more_black_brown_dolls_than_blonde_l1704_170449


namespace NUMINAMATH_GPT_num_of_chairs_per_row_l1704_170471

theorem num_of_chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) (chairs_per_row : ℕ)
  (h1 : total_chairs = 432)
  (h2 : num_rows = 27) :
  total_chairs = num_rows * chairs_per_row ↔ chairs_per_row = 16 :=
by
  sorry

end NUMINAMATH_GPT_num_of_chairs_per_row_l1704_170471


namespace NUMINAMATH_GPT_angus_tokens_count_l1704_170407

def worth_of_token : ℕ := 4
def elsa_tokens : ℕ := 60
def difference_worth : ℕ := 20

def elsa_worth : ℕ := elsa_tokens * worth_of_token
def angus_worth : ℕ := elsa_worth - difference_worth

def angus_tokens : ℕ := angus_worth / worth_of_token

theorem angus_tokens_count : angus_tokens = 55 := by
  sorry

end NUMINAMATH_GPT_angus_tokens_count_l1704_170407


namespace NUMINAMATH_GPT_S_range_l1704_170497

theorem S_range (x : ℝ) (y : ℝ) (S : ℝ) 
  (h1 : y = 2 * x - 1) 
  (h2 : 0 ≤ x) 
  (h3 : x ≤ 1 / 2) 
  (h4 : S = x * y) : 
  -1 / 8 ≤ S ∧ S ≤ 0 := 
sorry

end NUMINAMATH_GPT_S_range_l1704_170497


namespace NUMINAMATH_GPT_sin_diff_l1704_170430

theorem sin_diff (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1 / 3) 
  (h2 : Real.sin β - Real.cos α = 1 / 2) : 
  Real.sin (α - β) = -59 / 72 := 
sorry

end NUMINAMATH_GPT_sin_diff_l1704_170430


namespace NUMINAMATH_GPT_fraction_ordering_l1704_170425

theorem fraction_ordering : (4 / 17) < (6 / 25) ∧ (6 / 25) < (8 / 31) :=
by
  sorry

end NUMINAMATH_GPT_fraction_ordering_l1704_170425


namespace NUMINAMATH_GPT_problem_l1704_170484

variable {m n r t : ℚ}

theorem problem (h1 : m / n = 5 / 4) (h2 : r / t = 8 / 15) : (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1704_170484


namespace NUMINAMATH_GPT_smallest_divisor_of_2880_that_results_in_perfect_square_l1704_170487

theorem smallest_divisor_of_2880_that_results_in_perfect_square : 
  ∃ (n : ℕ), (n ∣ 2880) ∧ (∃ m : ℕ, 2880 / n = m * m) ∧ (∀ k : ℕ, (k ∣ 2880) ∧ (∃ m' : ℕ, 2880 / k = m' * m') → n ≤ k) ∧ n = 10 :=
sorry

end NUMINAMATH_GPT_smallest_divisor_of_2880_that_results_in_perfect_square_l1704_170487


namespace NUMINAMATH_GPT_power_division_identity_l1704_170496

theorem power_division_identity : 
  ∀ (a b c : ℕ), a = 3 → b = 12 → c = 2 → (3 ^ 12 / (3 ^ 2) ^ 2 = 6561) :=
by
  intros a b c h1 h2 h3
  sorry

end NUMINAMATH_GPT_power_division_identity_l1704_170496


namespace NUMINAMATH_GPT_point_B_coordinates_l1704_170424

theorem point_B_coordinates :
  ∃ (B : ℝ × ℝ), (B.1 < 0) ∧ (|B.2| = 4) ∧ (|B.1| = 5) ∧ (B = (-5, 4) ∨ B = (-5, -4)) :=
sorry

end NUMINAMATH_GPT_point_B_coordinates_l1704_170424


namespace NUMINAMATH_GPT_number_of_bags_proof_l1704_170482

def total_flight_time_hours : ℕ := 2
def minutes_per_hour : ℕ := 60
def total_minutes := total_flight_time_hours * minutes_per_hour

def peanuts_per_minute : ℕ := 1
def total_peanuts_eaten := total_minutes * peanuts_per_minute

def peanuts_per_bag : ℕ := 30
def number_of_bags : ℕ := total_peanuts_eaten / peanuts_per_bag

theorem number_of_bags_proof : number_of_bags = 4 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_number_of_bags_proof_l1704_170482


namespace NUMINAMATH_GPT_smarties_division_l1704_170463

theorem smarties_division (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := by
  sorry

end NUMINAMATH_GPT_smarties_division_l1704_170463


namespace NUMINAMATH_GPT_equal_sundays_tuesdays_days_l1704_170418

-- Define the problem in Lean
def num_equal_sundays_and_tuesdays_starts : ℕ :=
  3

-- Define a function that calculates the number of starting days that result in equal Sundays and Tuesdays
def calculate_sundays_tuesdays_starts (days_in_month : ℕ) : ℕ :=
  if days_in_month = 30 then 3 else 0

-- Prove that for a month of 30 days, there are 3 valid starting days for equal Sundays and Tuesdays
theorem equal_sundays_tuesdays_days :
  calculate_sundays_tuesdays_starts 30 = num_equal_sundays_and_tuesdays_starts :=
by 
  -- Proof outline here
  sorry

end NUMINAMATH_GPT_equal_sundays_tuesdays_days_l1704_170418


namespace NUMINAMATH_GPT_find_integers_satisfying_equation_l1704_170436

theorem find_integers_satisfying_equation :
  ∃ (a b c : ℤ), (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b = 1 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c = 1) ∨
                  (a = 2 ∧ b = -1 ∧ c = -1) ∨ (a = -1 ∧ b = 2 ∧ c = -1) ∨ (a = -1 ∧ b = -1 ∧ c = 2)
  ↔ (∃ (a b c : ℤ), 1 / 2 * (a + b) * (b + c) * (c + a) + (a + b + c) ^ 3 = 1 - a * b * c) := sorry

end NUMINAMATH_GPT_find_integers_satisfying_equation_l1704_170436


namespace NUMINAMATH_GPT_motorist_gallons_affordable_l1704_170409

-- Definitions based on the conditions in the problem
def expected_gallons : ℕ := 12
def actual_price_per_gallon : ℕ := 150
def price_difference : ℕ := 30
def expected_price_per_gallon : ℕ := actual_price_per_gallon - price_difference
def total_initial_cents : ℕ := expected_gallons * expected_price_per_gallon

-- Theorem stating that given the conditions, the motorist can afford 9 gallons of gas
theorem motorist_gallons_affordable : 
  total_initial_cents / actual_price_per_gallon = 9 := 
by
  sorry

end NUMINAMATH_GPT_motorist_gallons_affordable_l1704_170409


namespace NUMINAMATH_GPT_hyperbola_focus_l1704_170450

theorem hyperbola_focus :
  ∃ (x y : ℝ), 2 * x^2 - y^2 - 8 * x + 4 * y - 4 = 0 ∧ (x, y) = (2 + 2 * Real.sqrt 3, 2) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_hyperbola_focus_l1704_170450


namespace NUMINAMATH_GPT_y_intercept_of_line_l1704_170442

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) : y = -2 :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1704_170442


namespace NUMINAMATH_GPT_histogram_height_representation_l1704_170419

theorem histogram_height_representation (freq_ratio : ℝ) (frequency : ℝ) (class_interval : ℝ) 
  (H : freq_ratio = frequency / class_interval) : 
  freq_ratio = frequency / class_interval :=
by 
  sorry

end NUMINAMATH_GPT_histogram_height_representation_l1704_170419


namespace NUMINAMATH_GPT_ball_height_less_than_10_after_16_bounces_l1704_170490

noncomputable def bounce_height (initial : ℝ) (ratio : ℝ) (bounces : ℕ) : ℝ :=
  initial * ratio^bounces

theorem ball_height_less_than_10_after_16_bounces :
  let initial_height := 800
  let bounce_ratio := 3 / 4
  ∃ k : ℕ, k = 16 ∧ bounce_height initial_height bounce_ratio k < 10 := by
  let initial_height := 800
  let bounce_ratio := 3 / 4
  use 16
  sorry

end NUMINAMATH_GPT_ball_height_less_than_10_after_16_bounces_l1704_170490


namespace NUMINAMATH_GPT_base3_20121_to_base10_l1704_170445

def base3_to_base10 (n : ℕ) : ℕ :=
  2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0

theorem base3_20121_to_base10 :
  base3_to_base10 20121 = 178 :=
by
  sorry

end NUMINAMATH_GPT_base3_20121_to_base10_l1704_170445


namespace NUMINAMATH_GPT_table_price_l1704_170470

theorem table_price :
  ∃ C T : ℝ, (2 * C + T = 0.6 * (C + 2 * T)) ∧ (C + T = 72) ∧ (T = 63) :=
by
  sorry

end NUMINAMATH_GPT_table_price_l1704_170470


namespace NUMINAMATH_GPT_lines_through_point_l1704_170481

theorem lines_through_point {a b c : ℝ} :
  (3 = a + b) ∧ (3 = b + c) ∧ (3 = c + a) → (a = 1.5 ∧ b = 1.5 ∧ c = 1.5) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_lines_through_point_l1704_170481


namespace NUMINAMATH_GPT_find_b_l1704_170432

noncomputable def triangle_b_value (a : ℝ) (C : ℝ) (area : ℝ) : ℝ :=
  let sin_C := Real.sin C
  let b := (2 * area) / (a * sin_C)
  b

theorem find_b (h₁ : a = 1)
              (h₂ : C = Real.pi / 4)
              (h₃ : area = 2 * a) :
              triangle_b_value a C area = 8 * Real.sqrt 2 :=
by
  -- Definitions imply what we need
  sorry

end NUMINAMATH_GPT_find_b_l1704_170432


namespace NUMINAMATH_GPT_minimize_sum_of_squares_if_and_only_if_l1704_170440

noncomputable def minimize_sum_of_squares (a b c S : ℝ) (O : ℝ×ℝ×ℝ) (x y z : ℝ) : Prop :=
  let ax_by_cz := a * x + b * y + c * z
  ax_by_cz = 2 * S ∧
  x/y = a/b ∧
  y/z = b/c ∧
  x/z = a/c

theorem minimize_sum_of_squares_if_and_only_if (a b c S : ℝ) (O : ℝ×ℝ×ℝ) (x y z : ℝ) :
  (∃ P : ℝ, minimize_sum_of_squares a b c S O x y z) ↔ (x/y = a/b ∧ y/z = b/c ∧ x/z = a/c) := sorry

end NUMINAMATH_GPT_minimize_sum_of_squares_if_and_only_if_l1704_170440


namespace NUMINAMATH_GPT_salary_january_l1704_170417

variable (J F M A May : ℝ)

theorem salary_january 
  (h1 : J + F + M + A = 32000) 
  (h2 : F + M + A + May = 33600) 
  (h3 : May = 6500) : 
  J = 4900 := 
by {
 sorry 
}

end NUMINAMATH_GPT_salary_january_l1704_170417


namespace NUMINAMATH_GPT_complex_number_solution_l1704_170404

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i^2 = -1) (hz : i * (z - 1) = 1 - i) : z = -i :=
by sorry

end NUMINAMATH_GPT_complex_number_solution_l1704_170404


namespace NUMINAMATH_GPT_chess_club_officers_l1704_170451

/-- The Chess Club with 24 members needs to choose 3 officers: president,
    secretary, and treasurer. Each person can hold at most one office. 
    Alice and Bob will only serve together as officers. Prove that 
    the number of ways to choose the officers is 9372. -/
theorem chess_club_officers : 
  let members := 24
  let num_officers := 3
  let alice_and_bob_together := true
  ∃ n : ℕ, n = 9372 := sorry

end NUMINAMATH_GPT_chess_club_officers_l1704_170451


namespace NUMINAMATH_GPT_correct_option_d_l1704_170408

theorem correct_option_d (a b c : ℝ) (h: a < b ∧ b < 0) : a^2 > ab ∧ ab > b^2 :=
by
  sorry

end NUMINAMATH_GPT_correct_option_d_l1704_170408


namespace NUMINAMATH_GPT_trajectory_eq_l1704_170447

-- Define the conditions provided in the problem
def circle_eq (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * (m + 3) * x + 2 * (1 - 4 * m^2) + 16 * m^4 + 9 = 0

-- Define the required range for m based on the derivation
def m_valid (m : ℝ) : Prop :=
  -1/7 < m ∧ m < 1

-- Prove that the equation of the trajectory of the circle's center is y = 4(x-3)^2 -1 
-- and it's valid in the required range for x
theorem trajectory_eq (x y : ℝ) :
  (∃ m : ℝ, m_valid m ∧ y = 4 * (x - 3)^2 - 1 ∧ (x = m + 3) ∧ (y = 4 * m^2 - 1)) →
  y = 4 * (x - 3)^2 - 1 ∧ (20/7 < x) ∧ (x < 4) :=
by
  intro h
  cases' h with m hm
  sorry

end NUMINAMATH_GPT_trajectory_eq_l1704_170447


namespace NUMINAMATH_GPT_number_of_bananas_in_bowl_l1704_170467

theorem number_of_bananas_in_bowl (A P B : Nat) (h1 : P = A + 2) (h2 : B = P + 3) (h3 : A + P + B = 19) : B = 9 :=
sorry

end NUMINAMATH_GPT_number_of_bananas_in_bowl_l1704_170467


namespace NUMINAMATH_GPT_admin_staff_in_sample_l1704_170431

theorem admin_staff_in_sample (total_staff : ℕ) (admin_staff : ℕ) (total_samples : ℕ)
  (probability : ℚ) (h1 : total_staff = 200) (h2 : admin_staff = 24)
  (h3 : total_samples = 50) (h4 : probability = 50 / 200) :
  admin_staff * probability = 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_admin_staff_in_sample_l1704_170431


namespace NUMINAMATH_GPT_no_zeros_sin_log_l1704_170446

open Real

theorem no_zeros_sin_log (x : ℝ) (h1 : 1 < x) (h2 : x < exp 1) : ¬ (sin (log x) = 0) :=
sorry

end NUMINAMATH_GPT_no_zeros_sin_log_l1704_170446


namespace NUMINAMATH_GPT_rearrange_letters_no_adjacent_repeats_l1704_170474

-- Factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Problem conditions
def distinct_permutations (word : String) (freq_I : ℕ) (freq_L : ℕ) : ℕ :=
  factorial (String.length word) / (factorial freq_I * factorial freq_L)

-- No-adjacent-repeated permutations
def no_adjacent_repeats (word : String) (freq_I : ℕ) (freq_L : ℕ) : ℕ :=
  let total_permutations := distinct_permutations word freq_I freq_L
  let i_superletter_permutations := distinct_permutations (String.dropRight word 1) (freq_I - 1) freq_L
  let l_superletter_permutations := distinct_permutations (String.dropRight word 1) freq_I (freq_L - 1)
  let both_superletter_permutations := factorial (String.length word - 2)
  total_permutations - (i_superletter_permutations + l_superletter_permutations - both_superletter_permutations)

-- Given problem definition
def word := "BRILLIANT"
def freq_I := 2
def freq_L := 2

-- Proof problem statement
theorem rearrange_letters_no_adjacent_repeats :
  no_adjacent_repeats word freq_I freq_L = 55440 := by
  sorry

end NUMINAMATH_GPT_rearrange_letters_no_adjacent_repeats_l1704_170474


namespace NUMINAMATH_GPT_calculate_expression_l1704_170429

theorem calculate_expression (m : ℝ) : (-m)^2 * m^5 = m^7 := 
sorry

end NUMINAMATH_GPT_calculate_expression_l1704_170429


namespace NUMINAMATH_GPT_problem_statement_l1704_170493

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := λ x => f (x - 1)

theorem problem_statement :
  (∀ x : ℝ, f (-x) = f x) →  -- Condition: f is an even function.
  (∀ x : ℝ, g (-x) = -g x) → -- Condition: g is an odd function.
  (g 1 = 3) →                -- Condition: g passes through (1,3).
  (f 2012 + g 2013 = 6) :=   -- Statement to prove.
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1704_170493


namespace NUMINAMATH_GPT_half_percent_to_decimal_l1704_170456

def percent_to_decimal (x : ℚ) : ℚ := x / 100

theorem half_percent_to_decimal : percent_to_decimal (1 / 2) = 0.005 :=
by
  sorry

end NUMINAMATH_GPT_half_percent_to_decimal_l1704_170456


namespace NUMINAMATH_GPT_distance_between_houses_l1704_170420

-- Definitions
def speed : ℝ := 2          -- Amanda's speed in miles per hour
def time : ℝ := 3           -- Time taken by Amanda in hours

-- The theorem to prove distance is 6 miles
theorem distance_between_houses : speed * time = 6 := by
  sorry

end NUMINAMATH_GPT_distance_between_houses_l1704_170420


namespace NUMINAMATH_GPT_range_of_a_l1704_170483

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → x / (x^2 + 3 * x + 1) ≤ a) → a ≥ 1/5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1704_170483


namespace NUMINAMATH_GPT_exist_prime_not_dividing_l1704_170492

theorem exist_prime_not_dividing (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ ∀ n : ℕ, 0 < n → ¬ (q ∣ n^p - p) := 
sorry

end NUMINAMATH_GPT_exist_prime_not_dividing_l1704_170492


namespace NUMINAMATH_GPT_find_A_plus_B_l1704_170489

/-- Let A, B, C, and D be distinct digits such that 0 ≤ A, B, C, D ≤ 9.
    C and D are non-zero, and A ≠ B ≠ C ≠ D.
    If (A+B)/(C+D) is an integer and C+D is minimized,
    then prove that A + B = 15. -/
theorem find_A_plus_B
  (A B C D : ℕ)
  (h_digits : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_range : 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9)
  (h_nonzero_CD : C ≠ 0 ∧ D ≠ 0)
  (h_integer : (A + B) % (C + D) = 0)
  (h_min_CD : ∀ C' D', (C' ≠ C ∨ D' ≠ D) → (C' ≠ 0 ∧ D' ≠ 0 → (C + D ≤ C' + D'))) :
  A + B = 15 := 
sorry

end NUMINAMATH_GPT_find_A_plus_B_l1704_170489


namespace NUMINAMATH_GPT_worth_of_each_gold_bar_l1704_170414

theorem worth_of_each_gold_bar
  (rows : ℕ) (gold_bars_per_row : ℕ) (total_worth : ℕ)
  (h1 : rows = 4) (h2 : gold_bars_per_row = 20) (h3 : total_worth = 1600000)
  (total_gold_bars : ℕ) (h4 : total_gold_bars = rows * gold_bars_per_row) :
  total_worth / total_gold_bars = 20000 :=
by sorry

end NUMINAMATH_GPT_worth_of_each_gold_bar_l1704_170414


namespace NUMINAMATH_GPT_almond_butter_servings_l1704_170448

noncomputable def servings_in_container (total_tbsps : ℚ) (serving_size : ℚ) : ℚ :=
  total_tbsps / serving_size

theorem almond_butter_servings :
  servings_in_container (34 + 3/5) (5 + 1/2) = 6 + 21/55 :=
by
  sorry

end NUMINAMATH_GPT_almond_butter_servings_l1704_170448


namespace NUMINAMATH_GPT_polynomial_identity_l1704_170498

theorem polynomial_identity (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := by
  sorry

end NUMINAMATH_GPT_polynomial_identity_l1704_170498


namespace NUMINAMATH_GPT_roots_of_equation_l1704_170438

theorem roots_of_equation : ∃ x₁ x₂ : ℝ, (3 ^ x₁ = Real.log (x₁ + 9) / Real.log 3) ∧ 
                                     (3 ^ x₂ = Real.log (x₂ + 9) / Real.log 3) ∧ 
                                     (x₁ < 0) ∧ (x₂ > 0) := 
by {
  sorry
}

end NUMINAMATH_GPT_roots_of_equation_l1704_170438


namespace NUMINAMATH_GPT_probability_Rachel_Robert_in_picture_l1704_170461

noncomputable def Rachel_lap_time := 75
noncomputable def Robert_lap_time := 70
noncomputable def photo_time_start := 900
noncomputable def photo_time_end := 960
noncomputable def track_fraction := 1 / 5

theorem probability_Rachel_Robert_in_picture :
  let lap_time_Rachel := Rachel_lap_time
  let lap_time_Robert := Robert_lap_time
  let time_start := photo_time_start
  let time_end := photo_time_end
  let interval_Rachel := 15  -- ±15 seconds for Rachel
  let interval_Robert := 14  -- ±14 seconds for Robert
  let probability := (2 * interval_Robert) / (time_end - time_start) 
  probability = 7 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_Rachel_Robert_in_picture_l1704_170461


namespace NUMINAMATH_GPT_unique_solution_l1704_170499

-- Definitions of the problem
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

def satisfies_conditions (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime (4 * q - 1) ∧ (p + q) * (r - p) = p + r

theorem unique_solution (p q r : ℕ) (h : satisfies_conditions p q r) : (p, q, r) = (2, 3, 3) :=
  sorry

end NUMINAMATH_GPT_unique_solution_l1704_170499


namespace NUMINAMATH_GPT_tan_sum_l1704_170480

theorem tan_sum (x y : ℝ) (h1 : Real.sin x + Real.sin y = 85 / 65) (h2 : Real.cos x + Real.cos y = 60 / 65) :
  Real.tan x + Real.tan y = 17 / 12 :=
sorry

end NUMINAMATH_GPT_tan_sum_l1704_170480


namespace NUMINAMATH_GPT_simplify_correct_l1704_170468

open Polynomial

noncomputable def simplify_expression (y : ℚ) : Polynomial ℚ :=
  (3 * (Polynomial.C y) + 2) * (2 * (Polynomial.C y)^12 + 3 * (Polynomial.C y)^11 - (Polynomial.C y)^9 - (Polynomial.C y)^8)

theorem simplify_correct (y : ℚ) : 
  simplify_expression y = 6 * (Polynomial.C y)^13 + 13 * (Polynomial.C y)^12 + 6 * (Polynomial.C y)^11 - 3 * (Polynomial.C y)^10 - 5 * (Polynomial.C y)^9 - 2 * (Polynomial.C y)^8 := 
by 
  simp [simplify_expression]
  sorry

end NUMINAMATH_GPT_simplify_correct_l1704_170468


namespace NUMINAMATH_GPT_domain_of_f_l1704_170479

def domain_condition1 (x : ℝ) : Prop := 1 - |x - 1| > 0
def domain_condition2 (x : ℝ) : Prop := x - 1 ≠ 0

theorem domain_of_f :
  (∀ x : ℝ, domain_condition1 x ∧ domain_condition2 x → 0 < x ∧ x < 2 ∧ x ≠ 1) ↔
  (∀ x : ℝ, x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 2)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1704_170479


namespace NUMINAMATH_GPT_percentage_of_students_owning_cats_l1704_170460

theorem percentage_of_students_owning_cats (N C : ℕ) (hN : N = 500) (hC : C = 75) :
  (C / N : ℚ) * 100 = 15 := by
  sorry

end NUMINAMATH_GPT_percentage_of_students_owning_cats_l1704_170460


namespace NUMINAMATH_GPT_max_product_two_four_digit_numbers_l1704_170410

theorem max_product_two_four_digit_numbers :
  ∃ (a b : ℕ), 
    (a * b = max (8564 * 7321) (8531 * 7642)) 
    ∧ max 8531 8564 = 8531 ∧ 
    (∀ x y : ℕ, x * y ≤ 8531 * 7642 → x * y = max (8564 * 7321) (8531 * 7642)) :=
sorry

end NUMINAMATH_GPT_max_product_two_four_digit_numbers_l1704_170410


namespace NUMINAMATH_GPT_find_cows_l1704_170416

variable (D C : ℕ)

theorem find_cows (h1 : 2 * D + 4 * C = 2 * (D + C) + 36) : C = 18 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_cows_l1704_170416


namespace NUMINAMATH_GPT_vector_addition_result_l1704_170428

-- Definitions based on problem conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (y : ℝ) : ℝ × ℝ := (2, y)

-- The condition that vectors are parallel
def parallel_vectors (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2)

-- The main theorem to prove
theorem vector_addition_result (y : ℝ) (h : parallel_vectors vector_a (vector_b y)) : 
  (vector_a.1 + 2 * (vector_b y).1, vector_a.2 + 2 * (vector_b y).2) = (5, 10) :=
sorry

end NUMINAMATH_GPT_vector_addition_result_l1704_170428


namespace NUMINAMATH_GPT_karen_kept_cookies_l1704_170422

def total_cookies : ℕ := 50
def cookies_to_grandparents : ℕ := 8
def number_of_classmates : ℕ := 16
def cookies_per_classmate : ℕ := 2

theorem karen_kept_cookies (x : ℕ) 
  (H1 : x = total_cookies - (cookies_to_grandparents + number_of_classmates * cookies_per_classmate)) :
  x = 10 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_karen_kept_cookies_l1704_170422


namespace NUMINAMATH_GPT_determine_n_l1704_170411

-- Constants and variables
variables {a : ℕ → ℝ} {n : ℕ}

-- Definition for the condition at each vertex
def vertex_condition (a : ℕ → ℝ) (i : ℕ) : Prop :=
  a i = a (i - 1) * a (i + 1)

-- Mathematical problem statement
theorem determine_n (h : ∀ i, vertex_condition a i) (distinct_a : ∀ i j, a i ≠ a j) : n = 6 :=
sorry

end NUMINAMATH_GPT_determine_n_l1704_170411


namespace NUMINAMATH_GPT_avg_height_eq_61_l1704_170491

-- Define the constants and conditions
def Brixton : ℕ := 64
def Zara : ℕ := 64
def Zora := Brixton - 8
def Itzayana := Zora + 4

-- Define the total height of the four people
def total_height := Brixton + Zara + Zora + Itzayana

-- Define the average height
def average_height := total_height / 4

-- Theorem stating that the average height is 61 inches
theorem avg_height_eq_61 : average_height = 61 := by
  sorry

end NUMINAMATH_GPT_avg_height_eq_61_l1704_170491


namespace NUMINAMATH_GPT_right_triangle_circum_inradius_sum_l1704_170454

theorem right_triangle_circum_inradius_sum
  (a b : ℕ)
  (h1 : a = 16)
  (h2 : b = 30)
  (h_triangle : a^2 + b^2 = 34^2) :
  let c := 34
  let R := c / 2
  let A := a * b / 2
  let s := (a + b + c) / 2
  let r := A / s
  R + r = 23 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_circum_inradius_sum_l1704_170454


namespace NUMINAMATH_GPT_value_of_x_plus_y_l1704_170465

theorem value_of_x_plus_y (x y : ℝ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = 2) : x + y = 4/3 :=
sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l1704_170465


namespace NUMINAMATH_GPT_semicircle_area_difference_l1704_170437

theorem semicircle_area_difference 
  (A B C P D E F : Type) 
  (h₁ : S₅ - S₆ = 2) 
  (h₂ : S₁ - S₂ = 1) 
  : S₄ - S₃ = 3 :=
by
  -- Using Lean tactics to form the proof, place sorry for now.
  sorry

end NUMINAMATH_GPT_semicircle_area_difference_l1704_170437


namespace NUMINAMATH_GPT_max_value_g_l1704_170452

noncomputable def g (x : ℝ) := 4 * x - x ^ 4

theorem max_value_g : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 4 ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ Real.sqrt 4 → g y ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_value_g_l1704_170452


namespace NUMINAMATH_GPT_find_m_l1704_170459

theorem find_m (m : ℝ) (A B C D : ℝ × ℝ)
  (h1 : A = (m, 1)) (h2 : B = (-3, 4))
  (h3 : C = (0, 2)) (h4 : D = (1, 1))
  (h_parallel : (4 - 1) / (-3 - m) = (1 - 2) / (1 - 0)) :
  m = 0 :=
  by
  sorry

end NUMINAMATH_GPT_find_m_l1704_170459


namespace NUMINAMATH_GPT_find_number_l1704_170486

theorem find_number (x : ℝ) (h : x - (3/5) * x = 56) : x = 140 :=
sorry

end NUMINAMATH_GPT_find_number_l1704_170486


namespace NUMINAMATH_GPT_passing_percentage_correct_l1704_170400

-- The given conditions
def marks_obtained : ℕ := 175
def marks_failed : ℕ := 89
def max_marks : ℕ := 800

-- The theorem to prove
theorem passing_percentage_correct :
  (
    (marks_obtained + marks_failed : ℕ) * 100 / max_marks
  ) = 33 :=
sorry

end NUMINAMATH_GPT_passing_percentage_correct_l1704_170400


namespace NUMINAMATH_GPT_solution_to_equation_l1704_170433

noncomputable def solve_equation (x : ℝ) : Prop :=
  x + 2 = 1 / (x - 2) ∧ x ≠ 2

theorem solution_to_equation (x : ℝ) (h : solve_equation x) : x = Real.sqrt 5 ∨ x = -Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_solution_to_equation_l1704_170433


namespace NUMINAMATH_GPT_find_y_eq_54_div_23_l1704_170453

open BigOperators

theorem find_y_eq_54_div_23 (y : ℚ) (h : (Real.sqrt (8 * y) / Real.sqrt (6 * (y - 2))) = 3) : y = 54 / 23 := 
by
  sorry

end NUMINAMATH_GPT_find_y_eq_54_div_23_l1704_170453


namespace NUMINAMATH_GPT_percentage_increase_l1704_170401

theorem percentage_increase (old_earnings new_earnings : ℝ) (h_old : old_earnings = 50) (h_new : new_earnings = 70) :
  ((new_earnings - old_earnings) / old_earnings) * 100 = 40 :=
by
  rw [h_old, h_new]
  -- Simplification and calculation steps would go here
  sorry

end NUMINAMATH_GPT_percentage_increase_l1704_170401


namespace NUMINAMATH_GPT_probability_heart_spade_queen_l1704_170473

theorem probability_heart_spade_queen (h_cards : ℕ) (s_cards : ℕ) (q_cards : ℕ) (total_cards : ℕ) 
    (h_not_q : ℕ) (remaining_cards_after_2 : ℕ) (remaining_spades : ℕ) 
    (queen_remaining_after_2 : ℕ) (remaining_cards_after_1 : ℕ) :
    h_cards = 13 ∧ s_cards = 13 ∧ q_cards = 4 ∧ total_cards = 52 ∧ h_not_q = 12 ∧ remaining_cards_after_2 = 50 ∧
    remaining_spades = 13 ∧ queen_remaining_after_2 = 3 ∧ remaining_cards_after_1 = 51 →
    (h_cards / total_cards) * (remaining_spades / remaining_cards_after_1) * (q_cards / remaining_cards_after_2) + 
    (q_cards / total_cards) * (remaining_spades / remaining_cards_after_1) * (queen_remaining_after_2 / remaining_cards_after_2) = 
    221 / 44200 := by 
  sorry

end NUMINAMATH_GPT_probability_heart_spade_queen_l1704_170473


namespace NUMINAMATH_GPT_cauchy_problem_solution_l1704_170464

noncomputable def solution (y : ℝ → ℝ) (x : ℝ) : Prop :=
  y x = (x^2) / 2 + (x^3) / 6 + (x^4) / 12 + (x^5) / 20 + x + 1

theorem cauchy_problem_solution (y : ℝ → ℝ) (x : ℝ) 
  (h1: ∀ x, (deriv^[2] y) x = 1 + x + x^2 + x^3)
  (h2: y 0 = 1)
  (h3: deriv y 0 = 1) : 
  solution y x := 
by
  -- Proof Steps
  sorry

end NUMINAMATH_GPT_cauchy_problem_solution_l1704_170464


namespace NUMINAMATH_GPT_jungkook_age_l1704_170421

theorem jungkook_age
    (J U : ℕ)
    (h1 : J = U - 12)
    (h2 : (J + 3) + (U + 3) = 38) :
    J = 10 := 
sorry

end NUMINAMATH_GPT_jungkook_age_l1704_170421


namespace NUMINAMATH_GPT_prop1_prop2_l1704_170441

-- Proposition 1: Prove the contrapositive
theorem prop1 (q : ℝ) (h : ¬(∃ x : ℝ, x^2 + 2 * x + q = 0)) : q ≥ 1 :=
sorry

-- Proposition 2: Prove the contrapositive
theorem prop2 (x y : ℝ) (h : ¬(x = 0 ∧ y = 0)) : x^2 + y^2 ≠ 0 :=
sorry

end NUMINAMATH_GPT_prop1_prop2_l1704_170441


namespace NUMINAMATH_GPT_problem_statement_l1704_170458

open Real

noncomputable def log4 (x : ℝ) : ℝ := log x / log 4

noncomputable def a : ℝ := log4 (sqrt 5)
noncomputable def b : ℝ := log 2 / log 5
noncomputable def c : ℝ := log4 5

theorem problem_statement : b < a ∧ a < c :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1704_170458


namespace NUMINAMATH_GPT_nina_has_9_times_more_reading_homework_l1704_170434

theorem nina_has_9_times_more_reading_homework
  (ruby_math_homework : ℕ)
  (ruby_reading_homework : ℕ)
  (nina_total_homework : ℕ)
  (nina_math_homework_factor : ℕ)
  (h1 : ruby_math_homework = 6)
  (h2 : ruby_reading_homework = 2)
  (h3 : nina_total_homework = 48)
  (h4 : nina_math_homework_factor = 4) :
  nina_total_homework - (ruby_math_homework * (nina_math_homework_factor + 1)) = 9 * ruby_reading_homework := by
  sorry

end NUMINAMATH_GPT_nina_has_9_times_more_reading_homework_l1704_170434


namespace NUMINAMATH_GPT_express_x_n_prove_inequality_l1704_170485

variable (a b n : Real)
variable (x : ℕ → Real)

def trapezoid_conditions : Prop :=
  ∀ n, x 1 = a * b / (a + b) ∧ (x (n + 1) / x n = x (n + 1) / a)

theorem express_x_n (h : trapezoid_conditions a b x) : 
  ∀ n, x n = a * b / (a + n * b) := 
by
  sorry

theorem prove_inequality (h : trapezoid_conditions a b x) : 
  ∀ n, x n ≤ (a + n * b) / (4 * n) := 
by
  sorry

end NUMINAMATH_GPT_express_x_n_prove_inequality_l1704_170485


namespace NUMINAMATH_GPT_cos_60_eq_one_half_l1704_170413

theorem cos_60_eq_one_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_cos_60_eq_one_half_l1704_170413


namespace NUMINAMATH_GPT_line_eq_form_l1704_170457

def line_equation (x y : ℝ) : Prop :=
  ((3 : ℝ) * (x - 2) - (4 : ℝ) * (y + 3) = 0)

theorem line_eq_form (x y : ℝ) (h : line_equation x y) :
  ∃ (m b : ℝ), y = m * x + b ∧ (m = 3/4 ∧ b = -9/2) :=
by
  sorry

end NUMINAMATH_GPT_line_eq_form_l1704_170457


namespace NUMINAMATH_GPT_simple_interest_rate_l1704_170488

theorem simple_interest_rate (P : ℝ) (T : ℝ) (R : ℝ) (SI : ℝ) (hT : T = 8) 
  (hSI : SI = P / 5) : SI = (P * R * T) / 100 → R = 2.5 :=
by
  intro
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1704_170488


namespace NUMINAMATH_GPT_diameter_of_circumscribed_circle_l1704_170444

theorem diameter_of_circumscribed_circle (a : ℝ) (A : ℝ) (D : ℝ) 
  (h1 : a = 12) (h2 : A = 30) : D = 24 :=
by
  sorry

end NUMINAMATH_GPT_diameter_of_circumscribed_circle_l1704_170444


namespace NUMINAMATH_GPT_number_of_non_symmetric_letters_is_3_l1704_170494

def letters_in_JUNIOR : List Char := ['J', 'U', 'N', 'I', 'O', 'R']

def axis_of_symmetry (c : Char) : Bool :=
  match c with
  | 'J' => false
  | 'U' => true
  | 'N' => false
  | 'I' => true
  | 'O' => true
  | 'R' => false
  | _   => false

def letters_with_no_symmetry : List Char :=
  letters_in_JUNIOR.filter (λ c => ¬axis_of_symmetry c)

theorem number_of_non_symmetric_letters_is_3 :
  letters_with_no_symmetry.length = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_non_symmetric_letters_is_3_l1704_170494


namespace NUMINAMATH_GPT_base_eight_to_base_ten_642_l1704_170472

theorem base_eight_to_base_ten_642 :
  let d0 := 2
  let d1 := 4
  let d2 := 6
  let base := 8
  d0 * base^0 + d1 * base^1 + d2 * base^2 = 418 := 
by
  sorry

end NUMINAMATH_GPT_base_eight_to_base_ten_642_l1704_170472


namespace NUMINAMATH_GPT_sum_infinite_series_eq_l1704_170415

theorem sum_infinite_series_eq {x : ℝ} (hx : |x| < 1) :
  (∑' n : ℕ, (n + 1) * x^n) = 1 / (1 - x)^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_infinite_series_eq_l1704_170415


namespace NUMINAMATH_GPT_find_k_from_given_solution_find_other_root_l1704_170435

-- Given
def one_solution_of_first_eq_is_same_as_second (x k : ℝ) : Prop :=
  x^2 + k * x - 2 = 0 ∧ (x + 1) / (x - 1) = 3

-- To find k
theorem find_k_from_given_solution : ∃ k : ℝ, ∃ x : ℝ, one_solution_of_first_eq_is_same_as_second x k ∧ k = -1 := by
  sorry

-- To find the other root
theorem find_other_root : ∃ x2 : ℝ, (x2 = -1) := by
  sorry

end NUMINAMATH_GPT_find_k_from_given_solution_find_other_root_l1704_170435


namespace NUMINAMATH_GPT_number_of_boys_is_320_l1704_170475

-- Definition of the problem's conditions
variable (B G : ℕ)
axiom condition1 : B + G = 400
axiom condition2 : G = (B / 400) * 100

-- Stating the theorem to prove number of boys is 320
theorem number_of_boys_is_320 : B = 320 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_is_320_l1704_170475


namespace NUMINAMATH_GPT_min_value_of_sum_of_squares_l1704_170443

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x * y + y * z + x * z = 4) :
  x^2 + y^2 + z^2 ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_sum_of_squares_l1704_170443
