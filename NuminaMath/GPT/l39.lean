import Mathlib

namespace NUMINAMATH_GPT_Berry_Temperature_Friday_l39_3922

theorem Berry_Temperature_Friday (temps : Fin 6 → ℝ) (avg_temp : ℝ) (total_days : ℕ) (friday_temp : ℝ) :
  temps 0 = 99.1 → 
  temps 1 = 98.2 →
  temps 2 = 98.7 →
  temps 3 = 99.3 →
  temps 4 = 99.8 →
  temps 5 = 98.9 →
  avg_temp = 99 →
  total_days = 7 →
  friday_temp = (avg_temp * total_days) - (temps 0 + temps 1 + temps 2 + temps 3 + temps 4 + temps 5) →
  friday_temp = 99 :=
by 
  intros h0 h1 h2 h3 h4 h5 h_avg h_days h_friday
  sorry

end NUMINAMATH_GPT_Berry_Temperature_Friday_l39_3922


namespace NUMINAMATH_GPT_average_words_per_minute_l39_3910

theorem average_words_per_minute 
  (total_words : ℕ) 
  (total_hours : ℕ) 
  (h_words : total_words = 30000) 
  (h_hours : total_hours = 100) : 
  (total_words / total_hours / 60 = 5) := by
  sorry

end NUMINAMATH_GPT_average_words_per_minute_l39_3910


namespace NUMINAMATH_GPT_all_propositions_imply_l39_3993

variables (p q r : Prop)

theorem all_propositions_imply (hpqr : p ∧ q ∧ r)
                               (hnpqr : ¬p ∧ q ∧ ¬r)
                               (hpnqr : p ∧ ¬q ∧ r)
                               (hnpnqr : ¬p ∧ ¬q ∧ ¬r) :
  (p → q) ∨ r :=
by { sorry }

end NUMINAMATH_GPT_all_propositions_imply_l39_3993


namespace NUMINAMATH_GPT_domain_range_sum_l39_3932

def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem domain_range_sum (m n : ℝ) (hmn : ∀ x, m ≤ x ∧ x ≤ n → (f x = 3 * x)) : m + n = -1 :=
by
  sorry

end NUMINAMATH_GPT_domain_range_sum_l39_3932


namespace NUMINAMATH_GPT_points_lie_on_hyperbola_l39_3981

def point_on_hyperbola (t : ℝ) : Prop :=
  let x := 2 * Real.exp t - 2 * Real.exp (-t)
  let y := 4 * (Real.exp t + Real.exp (-t))
  (y^2) / 16 - (x^2) / 4 = 1

theorem points_lie_on_hyperbola : ∀ t : ℝ, point_on_hyperbola t :=
by
  intro t
  sorry

end NUMINAMATH_GPT_points_lie_on_hyperbola_l39_3981


namespace NUMINAMATH_GPT_square_area_l39_3984

theorem square_area 
  (s r l : ℝ)
  (h_r_s : r = s)
  (h_l_r : l = (2/5) * r)
  (h_area_rect : l * 10 = 120) : 
  s^2 = 900 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_square_area_l39_3984


namespace NUMINAMATH_GPT_average_birth_rate_l39_3982

theorem average_birth_rate (B : ℕ) 
  (death_rate : ℕ := 3)
  (daily_net_increase : ℕ := 86400) 
  (intervals_per_day : ℕ := 86400 / 2) 
  (net_increase : ℕ := (B - death_rate) * intervals_per_day) : 
  net_increase = daily_net_increase → 
  B = 5 := 
sorry

end NUMINAMATH_GPT_average_birth_rate_l39_3982


namespace NUMINAMATH_GPT_infinite_series_sum_l39_3992

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l39_3992


namespace NUMINAMATH_GPT_fraction_multiplication_l39_3974

theorem fraction_multiplication : (1 / 3) * (1 / 4) * (1 / 5) * 60 = 1 := by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_l39_3974


namespace NUMINAMATH_GPT_sindbad_can_identify_eight_genuine_dinars_l39_3918

/--
Sindbad has 11 visually identical dinars in his purse, one of which may be counterfeit and differs in weight from the genuine ones. Using a balance scale twice without weights, it's possible to identify at least 8 genuine dinars.
-/
theorem sindbad_can_identify_eight_genuine_dinars (dinars : Fin 11 → ℝ) (is_genuine : Fin 11 → Prop) :
  (∃! i, ¬ is_genuine i) → 
  (∃ S : Finset (Fin 11), S.card = 8 ∧ S ⊆ (Finset.univ : Finset (Fin 11)) ∧ ∀ i ∈ S, is_genuine i) :=
sorry

end NUMINAMATH_GPT_sindbad_can_identify_eight_genuine_dinars_l39_3918


namespace NUMINAMATH_GPT_car_circuit_velocity_solution_l39_3999

theorem car_circuit_velocity_solution
    (v_s v_p v_d : ℕ)
    (h1 : v_s < v_p)
    (h2 : v_p < v_d)
    (h3 : s = d)
    (h4 : s + p + d = 600)
    (h5 : (d : ℚ) / v_s + (p : ℚ) / v_p + (d : ℚ) / v_d = 50) :
    (v_s = 7 ∧ v_p = 12 ∧ v_d = 42) ∨
    (v_s = 8 ∧ v_p = 12 ∧ v_d = 24) ∨
    (v_s = 9 ∧ v_p = 12 ∧ v_d = 18) ∨
    (v_s = 10 ∧ v_p = 12 ∧ v_d = 15) :=
by
  sorry

end NUMINAMATH_GPT_car_circuit_velocity_solution_l39_3999


namespace NUMINAMATH_GPT_probability_at_least_one_woman_l39_3983

theorem probability_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ)
  (h1 : total_people = 10) (h2 : men = 5) (h3 : women = 5) (h4 : selected = 3) :
  (1 - (men / total_people) * ((men - 1) / (total_people - 1)) * ((men - 2) / (total_people - 2))) = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_woman_l39_3983


namespace NUMINAMATH_GPT_largest_alpha_l39_3976

theorem largest_alpha (a b : ℕ) (h1 : a < b) (h2 : b < 2 * a) (N : ℕ) :
  ∃ (α : ℝ), α = 1 / (2 * a^2 - 2 * a * b + b^2) ∧
  (∃ marked_cells : ℕ, marked_cells ≥ α * (N:ℝ)^2) :=
by
  sorry

end NUMINAMATH_GPT_largest_alpha_l39_3976


namespace NUMINAMATH_GPT_Eric_eggs_collected_l39_3953

theorem Eric_eggs_collected : 
  (∀ (chickens : ℕ) (eggs_per_chicken_per_day : ℕ) (days : ℕ),
    chickens = 4 ∧ eggs_per_chicken_per_day = 3 ∧ days = 3 → 
    chickens * eggs_per_chicken_per_day * days = 36) :=
by
  sorry

end NUMINAMATH_GPT_Eric_eggs_collected_l39_3953


namespace NUMINAMATH_GPT_num_three_digit_integers_with_zero_in_units_place_divisible_by_30_l39_3989

noncomputable def countThreeDigitMultiplesOf30WithZeroInUnitsPlace : ℕ :=
  let a := 120
  let d := 30
  let l := 990
  (l - a) / d + 1

theorem num_three_digit_integers_with_zero_in_units_place_divisible_by_30 :
  countThreeDigitMultiplesOf30WithZeroInUnitsPlace = 30 := by
  sorry

end NUMINAMATH_GPT_num_three_digit_integers_with_zero_in_units_place_divisible_by_30_l39_3989


namespace NUMINAMATH_GPT_rational_solutions_equation_l39_3952

theorem rational_solutions_equation :
  ∃ x : ℚ, (|x - 19| + |x - 93| = 74 ∧ x ∈ {y : ℚ | 19 ≤ y ∨ 19 < y ∧ y < 93 ∨ y ≥ 93}) :=
sorry

end NUMINAMATH_GPT_rational_solutions_equation_l39_3952


namespace NUMINAMATH_GPT_carousel_ratio_l39_3928

theorem carousel_ratio (P : ℕ) (h : 3 + P + 2*P + P/3 = 33) : P / 3 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_carousel_ratio_l39_3928


namespace NUMINAMATH_GPT_triangle_inradius_is_2_5_l39_3967

variable (A : ℝ) (p : ℝ) (r : ℝ)

def triangle_has_given_inradius (A p : ℝ) : Prop :=
  A = r * p / 2

theorem triangle_inradius_is_2_5 (h₁ : A = 25) (h₂ : p = 20) :
  triangle_has_given_inradius A p r → r = 2.5 := sorry

end NUMINAMATH_GPT_triangle_inradius_is_2_5_l39_3967


namespace NUMINAMATH_GPT_simplify_expression_l39_3905

theorem simplify_expression (a : ℝ) : a^2 * (-a)^4 = a^6 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l39_3905


namespace NUMINAMATH_GPT_solution_of_inequality_l39_3979

theorem solution_of_inequality (x : ℝ) : x * (x - 1) < 2 ↔ -1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_GPT_solution_of_inequality_l39_3979


namespace NUMINAMATH_GPT_both_pumps_drain_lake_l39_3933

theorem both_pumps_drain_lake (T : ℝ) (h₁ : 1 / 9 + 1 / 6 = 5 / 18) : 
  (5 / 18) * T = 1 → T = 18 / 5 := sorry

end NUMINAMATH_GPT_both_pumps_drain_lake_l39_3933


namespace NUMINAMATH_GPT_arith_seq_general_formula_geom_seq_sum_l39_3947

-- Problem 1
theorem arith_seq_general_formula (a : ℕ → ℕ) (d : ℕ) (h_d : d = 3) (h_a1 : a 1 = 4) :
  a n = 3 * n + 1 :=
sorry

-- Problem 2
theorem geom_seq_sum (b : ℕ → ℚ) (S : ℕ → ℚ) (h_b1 : b 1 = 1 / 3) (r : ℚ) (h_r : r = 1 / 3) :
  S n = (1 / 2) * (1 - (1 / 3 ^ n)) :=
sorry

end NUMINAMATH_GPT_arith_seq_general_formula_geom_seq_sum_l39_3947


namespace NUMINAMATH_GPT_margie_change_l39_3963

theorem margie_change : 
  let cost_per_apple := 0.30
  let cost_per_orange := 0.40
  let number_of_apples := 5
  let number_of_oranges := 4
  let total_money := 10.00
  let total_cost_of_apples := cost_per_apple * number_of_apples
  let total_cost_of_oranges := cost_per_orange * number_of_oranges
  let total_cost_of_fruits := total_cost_of_apples + total_cost_of_oranges
  let change_received := total_money - total_cost_of_fruits
  change_received = 6.90 :=
by
  sorry

end NUMINAMATH_GPT_margie_change_l39_3963


namespace NUMINAMATH_GPT_problem_statement_l39_3912

theorem problem_statement (p : ℝ) : 
  (∀ (q : ℝ), q > 0 → (3 * (p * q^2 + 2 * p^2 * q + 2 * q^2 + 5 * p * q)) / (p + q) > 3 * p^2 * q) 
  ↔ (0 ≤ p ∧ p ≤ 7.275) :=
sorry

end NUMINAMATH_GPT_problem_statement_l39_3912


namespace NUMINAMATH_GPT_annual_income_correct_l39_3949

-- Define the principal amounts and interest rates
def principal_1 : ℝ := 3000
def rate_1 : ℝ := 0.085

def principal_2 : ℝ := 5000
def rate_2 : ℝ := 0.064

-- Define the interest calculations for each investment
def interest_1 : ℝ := principal_1 * rate_1
def interest_2 : ℝ := principal_2 * rate_2

-- Define the total annual income
def total_annual_income : ℝ := interest_1 + interest_2

-- Proof statement
theorem annual_income_correct : total_annual_income = 575 :=
by
  sorry

end NUMINAMATH_GPT_annual_income_correct_l39_3949


namespace NUMINAMATH_GPT_find_base_l39_3973

noncomputable def log_base (a x : ℝ) := Real.log x / Real.log a

theorem find_base (a : ℝ) (h : 1 < a) :
  (log_base a (2 * a) - log_base a a = 1 / 2) → a = 4 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_find_base_l39_3973


namespace NUMINAMATH_GPT_system_of_equations_solution_l39_3957

theorem system_of_equations_solution (x y : ℚ) :
  (3 * x^2 + 2 * y^2 + 2 * x + 3 * y = 0 ∧ 4 * x^2 - 3 * y^2 - 3 * x + 4 * y = 0) ↔ 
  ((x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1)) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l39_3957


namespace NUMINAMATH_GPT_max_problems_missed_to_pass_l39_3950

theorem max_problems_missed_to_pass (total_problems : ℕ) (min_percentage : ℚ) 
  (h_total_problems : total_problems = 40) 
  (h_min_percentage : min_percentage = 0.85) : 
  ∃ max_missed : ℕ, max_missed = total_problems - ⌈total_problems * min_percentage⌉₊ ∧ max_missed = 6 :=
by
  sorry

end NUMINAMATH_GPT_max_problems_missed_to_pass_l39_3950


namespace NUMINAMATH_GPT_largest_integer_solution_l39_3935

theorem largest_integer_solution (x : ℤ) (h : -x ≥ 2 * x + 3) : x ≤ -1 := sorry

end NUMINAMATH_GPT_largest_integer_solution_l39_3935


namespace NUMINAMATH_GPT_min_episodes_to_watch_l39_3909

theorem min_episodes_to_watch (T W H F Sa Su M trip_days total_episodes: ℕ)
  (hW: W = 1) (hTh: H = 1) (hF: F = 1) (hSa: Sa = 2) (hSu: Su = 2) (hMo: M = 0)
  (total_episodes_eq: total_episodes = 60)
  (trip_days_eq: trip_days = 17):
  total_episodes - ((4 * W + 2 * Sa + 1 * M) * (trip_days / 7) + (trip_days % 7) * (W + Sa + Su + Mo)) = 39 := 
by
  sorry

end NUMINAMATH_GPT_min_episodes_to_watch_l39_3909


namespace NUMINAMATH_GPT_train_speed_ratio_l39_3927

theorem train_speed_ratio 
  (distance_2nd_train : ℕ)
  (time_2nd_train : ℕ)
  (speed_1st_train : ℚ)
  (H1 : distance_2nd_train = 400)
  (H2 : time_2nd_train = 4)
  (H3 : speed_1st_train = 87.5) :
  distance_2nd_train / time_2nd_train = 100 ∧ 
  (speed_1st_train / (distance_2nd_train / time_2nd_train)) = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_ratio_l39_3927


namespace NUMINAMATH_GPT_thomas_blocks_total_l39_3938

theorem thomas_blocks_total 
  (first_stack : ℕ)
  (second_stack : ℕ)
  (third_stack : ℕ)
  (fourth_stack : ℕ)
  (fifth_stack : ℕ) 
  (h1 : first_stack = 7)
  (h2 : second_stack = first_stack + 3)
  (h3 : third_stack = second_stack - 6)
  (h4 : fourth_stack = third_stack + 10)
  (h5 : fifth_stack = 2 * second_stack) :
  first_stack + second_stack + third_stack + fourth_stack + fifth_stack = 55 :=
by
  sorry

end NUMINAMATH_GPT_thomas_blocks_total_l39_3938


namespace NUMINAMATH_GPT_playground_total_l39_3923

def boys : ℕ := 44
def girls : ℕ := 53

theorem playground_total : boys + girls = 97 := by
  sorry

end NUMINAMATH_GPT_playground_total_l39_3923


namespace NUMINAMATH_GPT_divides_quartic_sum_l39_3934

theorem divides_quartic_sum (a b c n : ℤ) (h1 : n ∣ (a + b + c)) (h2 : n ∣ (a^2 + b^2 + c^2)) : n ∣ (a^4 + b^4 + c^4) := 
sorry

end NUMINAMATH_GPT_divides_quartic_sum_l39_3934


namespace NUMINAMATH_GPT_f_g_5_l39_3985

def g (x : ℕ) : ℕ := 4 * x + 10

def f (x : ℕ) : ℕ := 6 * x - 12

theorem f_g_5 : f (g 5) = 168 := by
  sorry

end NUMINAMATH_GPT_f_g_5_l39_3985


namespace NUMINAMATH_GPT_optimal_ticket_price_l39_3977

noncomputable def revenue (x : ℕ) : ℤ :=
  if x < 6 then -5750
  else if x ≤ 10 then 1000 * (x : ℤ) - 5750
  else if x ≤ 38 then -30 * (x : ℤ)^2 + 1300 * (x : ℤ) - 5750
  else -5750

theorem optimal_ticket_price :
  revenue 22 = 8330 :=
by
  sorry

end NUMINAMATH_GPT_optimal_ticket_price_l39_3977


namespace NUMINAMATH_GPT_max_value_of_function_l39_3939

noncomputable def f (x : ℝ) (α : ℝ) := x ^ α

theorem max_value_of_function (α : ℝ)
  (h₁ : f 4 α = 2)
  : ∃ a : ℝ, 3 ≤ a ∧ a ≤ 5 ∧ (f (a - 3) (α) + f (5 - a) α = 2) := 
sorry

end NUMINAMATH_GPT_max_value_of_function_l39_3939


namespace NUMINAMATH_GPT_factorize_1_factorize_2_factorize_3_solve_system_l39_3955

-- Proving the factorization identities
theorem factorize_1 (y : ℝ) : 5 * y - 10 * y^2 = 5 * y * (1 - 2 * y) :=
by
  sorry

theorem factorize_2 (m : ℝ) : (3 * m - 1)^2 - 9 = (3 * m + 2) * (3 * m - 4) :=
by
  sorry

theorem factorize_3 (a b : ℝ) : a^2 * b - 4 * a * b + 4 * b = b * (a - 2)^2 :=
by
  sorry

-- Proving the solution to the system of equations
theorem solve_system (x y : ℝ) (h1 : x - y = 3) (h2 : x - 3 * y = -1) : x = 5 ∧ y = 2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_1_factorize_2_factorize_3_solve_system_l39_3955


namespace NUMINAMATH_GPT_option_B_shares_asymptotes_l39_3961

-- Define the given hyperbola equation
def given_hyperbola (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1

-- The asymptotes for the given hyperbola
def asymptotes_of_given_hyperbola (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Define the hyperbola for option B
def option_B_hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 16) = 1

-- The asymptotes for option B hyperbola
def asymptotes_of_option_B_hyperbola (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Theorem stating that the hyperbola in option B shares the same asymptotes as the given hyperbola
theorem option_B_shares_asymptotes :
  (∀ x y : ℝ, given_hyperbola x y → asymptotes_of_given_hyperbola x y) →
  (∀ x y : ℝ, option_B_hyperbola x y → asymptotes_of_option_B_hyperbola x y) :=
by
  intros h₁ h₂
  -- Here should be the proof to show they have the same asymptotes
  sorry

end NUMINAMATH_GPT_option_B_shares_asymptotes_l39_3961


namespace NUMINAMATH_GPT_page_number_added_twice_l39_3978

theorem page_number_added_twice (n p : ℕ) (Hn : 1 ≤ n) (Hsum : (n * (n + 1)) / 2 + p = 2630) : 
  p = 2 :=
sorry

end NUMINAMATH_GPT_page_number_added_twice_l39_3978


namespace NUMINAMATH_GPT_min_odd_is_1_l39_3969

def min_odd_integers (a b c d e f : ℤ) : ℤ :=
  if (a + b) % 2 = 0 ∧ 
     (a + b + c + d) % 2 = 1 ∧ 
     (a + b + c + d + e + f) % 2 = 0 then
    1
  else
    sorry -- This should be replaced by a calculation of the true minimum based on conditions.

def satisfies_conditions (a b c d e f : ℤ) :=
  a + b = 30 ∧ 
  a + b + c + d = 47 ∧ 
  a + b + c + d + e + f = 65

theorem min_odd_is_1 (a b c d e f : ℤ) (h : satisfies_conditions a b c d e f) : 
  min_odd_integers a b c d e f = 1 := 
sorry

end NUMINAMATH_GPT_min_odd_is_1_l39_3969


namespace NUMINAMATH_GPT_solution_l39_3926

noncomputable def problem_statement (x y : ℝ) (hx : x > 1) (hy : y > 1) (h : (Real.log x / Real.log 3)^4 + (Real.log y / Real.log 5)^4 + 16 = 12 * (Real.log x / Real.log 3) * (Real.log y / Real.log 5)) : ℝ :=
  (x^2 * y^2)

theorem solution : ∀ x y : ℝ, x > 1 → y > 1 → (Real.log x / Real.log 3)^4 + (Real.log y / Real.log 5)^4 + 16 = 12 * (Real.log x / Real.log 3) * (Real.log y / Real.log 5) →
  (x^2 * y^2) = 225^(Real.sqrt 2) :=
by
  intros x y hx hy h
  sorry

end NUMINAMATH_GPT_solution_l39_3926


namespace NUMINAMATH_GPT_distance_between_stations_l39_3901

-- Definitions based on conditions in step a):
def speed_train1 : ℝ := 20  -- speed of the first train in km/hr
def speed_train2 : ℝ := 25  -- speed of the second train in km/hr
def extra_distance : ℝ := 55  -- one train has traveled 55 km more

-- Definition of the proof problem
theorem distance_between_stations :
  ∃ D1 D2 T : ℝ, D1 = speed_train1 * T ∧ D2 = speed_train2 * T ∧ D2 = D1 + extra_distance ∧ D1 + D2 = 495 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_stations_l39_3901


namespace NUMINAMATH_GPT_total_liquid_poured_out_l39_3942

noncomputable def capacity1 := 2
noncomputable def capacity2 := 6
noncomputable def percentAlcohol1 := 0.3
noncomputable def percentAlcohol2 := 0.4
noncomputable def totalCapacity := 10
noncomputable def finalConcentration := 0.3

theorem total_liquid_poured_out :
  capacity1 + capacity2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_total_liquid_poured_out_l39_3942


namespace NUMINAMATH_GPT_smallest_fraction_gt_five_sevenths_l39_3998

theorem smallest_fraction_gt_five_sevenths (a b : ℕ) (h1 : 10 ≤ a ∧ a ≤ 99) (h2 : 10 ≤ b ∧ b ≤ 99) (h3 : 7 * a > 5 * b) : a = 68 ∧ b = 95 :=
sorry

end NUMINAMATH_GPT_smallest_fraction_gt_five_sevenths_l39_3998


namespace NUMINAMATH_GPT_johns_share_l39_3990

theorem johns_share
  (total_amount : ℕ)
  (ratio_john : ℕ)
  (ratio_jose : ℕ)
  (ratio_binoy : ℕ)
  (total_parts : ℕ)
  (value_per_part : ℕ)
  (johns_parts : ℕ)
  (johns_share : ℕ)
  (h1 : total_amount = 4800)
  (h2 : ratio_john = 2)
  (h3 : ratio_jose = 4)
  (h4 : ratio_binoy = 6)
  (h5 : total_parts = ratio_john + ratio_jose + ratio_binoy)
  (h6 : value_per_part = total_amount / total_parts)
  (h7 : johns_parts = ratio_john)
  (h8 : johns_share = value_per_part * johns_parts) :
  johns_share = 800 := by
  sorry

end NUMINAMATH_GPT_johns_share_l39_3990


namespace NUMINAMATH_GPT_jamie_nickels_l39_3904

theorem jamie_nickels (x : ℕ) (hx : 5 * x + 10 * x + 25 * x = 1320) : x = 33 :=
sorry

end NUMINAMATH_GPT_jamie_nickels_l39_3904


namespace NUMINAMATH_GPT_sum_reciprocals_factors_12_l39_3925

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end NUMINAMATH_GPT_sum_reciprocals_factors_12_l39_3925


namespace NUMINAMATH_GPT_car_rental_cost_l39_3959

theorem car_rental_cost (daily_rent : ℕ) (rent_duration : ℕ) (mileage_rate : ℚ) (mileage : ℕ) (total_cost : ℕ) :
  daily_rent = 30 → rent_duration = 5 → mileage_rate = 0.25 → mileage = 500 → total_cost = 275 :=
by
  intros hd hr hm hl
  sorry

end NUMINAMATH_GPT_car_rental_cost_l39_3959


namespace NUMINAMATH_GPT_roots_order_l39_3980

theorem roots_order {a b m n : ℝ} (h1 : m < n) (h2 : a < b)
  (hm : 1 - (m - a) * (m - b) = 0) (hn : 1 - (n - a) * (n - b) = 0) :
  m < a ∧ a < b ∧ b < n :=
sorry

end NUMINAMATH_GPT_roots_order_l39_3980


namespace NUMINAMATH_GPT_contradiction_proof_l39_3915

theorem contradiction_proof (a b c d : ℝ) (h1 : a = 1) (h2 : b = 1) (h3 : c = 1) (h4 : d = 1) (h5 : a * c + b * d > 1) : ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
by 
  sorry

end NUMINAMATH_GPT_contradiction_proof_l39_3915


namespace NUMINAMATH_GPT_problem_statement_l39_3986

theorem problem_statement
  (f : ℝ → ℝ)
  (h0 : ∀ x, 0 <= x → x <= 1 → 0 <= f x)
  (h1 : ∀ x y, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 
        (f x + f y) / 2 ≤ f ((x + y) / 2) + 1) :
  ∀ (u v w : ℝ), 
    0 ≤ u ∧ u < v ∧ v < w ∧ w ≤ 1 → 
    (w - v) / (w - u) * f u + (v - u) / (w - u) * f w ≤ f v + 2 :=
by
  intros u v w h
  sorry

end NUMINAMATH_GPT_problem_statement_l39_3986


namespace NUMINAMATH_GPT_correct_article_usage_l39_3965

def sentence : String :=
  "While he was at ____ college, he took part in the march, and was soon thrown into ____ prison."

def rules_for_articles (context : String) (noun : String) : String → Bool
| "the" => noun ≠ "college" ∨ context = "specific"
| ""    => noun = "college" ∨ noun = "prison"
| _     => false

theorem correct_article_usage : 
  rules_for_articles "general" "college" "" ∧ 
  rules_for_articles "general" "prison" "" :=
by
  sorry

end NUMINAMATH_GPT_correct_article_usage_l39_3965


namespace NUMINAMATH_GPT_symmetric_point_l39_3968

theorem symmetric_point (x0 y0 : ℝ) (P : ℝ × ℝ) (line : ℝ → ℝ) 
  (hP : P = (-1, 3)) (hline : ∀ x, line x = x) :
  ((x0, y0) = (3, -1)) ↔
    ( ∃ M : ℝ × ℝ, M = ((x0 - -1) / 2, (y0 + 3) / 2) ∧ M.1 = M.2 ) ∧ 
    ( ∃ l : ℝ, l = (y0 - 3) / (x0 + 1) ∧ l = -1 ) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_l39_3968


namespace NUMINAMATH_GPT_collinear_points_b_value_l39_3940

theorem collinear_points_b_value (b : ℝ)
    (h : let slope1 := (4 - (-6)) / ((2 * b + 1) - 4)
         let slope2 := (1 - (-6)) / ((-3 * b + 2) - 4)
         slope1 = slope2) :
    b = -1 / 44 :=
by
  have slope1 := (4 - (-6)) / ((2 * b + 1) - 4)
  have slope2 := (1 - (-6)) / ((-3 * b + 2) - 4)
  have := h
  sorry

end NUMINAMATH_GPT_collinear_points_b_value_l39_3940


namespace NUMINAMATH_GPT_speed_of_second_car_l39_3907

theorem speed_of_second_car
  (t : ℝ) (d : ℝ) (d1 : ℝ) (d2 : ℝ) (v : ℝ)
  (h1 : t = 2.5)
  (h2 : d = 175)
  (h3 : d1 = 25 * t)
  (h4 : d2 = v * t)
  (h5 : d1 + d2 = d) :
  v = 45 := by sorry

end NUMINAMATH_GPT_speed_of_second_car_l39_3907


namespace NUMINAMATH_GPT_nancy_more_money_l39_3900

def jade_available := 1920
def giraffe_jade := 120
def elephant_jade := 2 * giraffe_jade
def giraffe_price := 150
def elephant_price := 350

def num_giraffes := jade_available / giraffe_jade
def num_elephants := jade_available / elephant_jade
def revenue_giraffes := num_giraffes * giraffe_price
def revenue_elephants := num_elephants * elephant_price

def revenue_difference := revenue_elephants - revenue_giraffes

theorem nancy_more_money : revenue_difference = 400 :=
by sorry

end NUMINAMATH_GPT_nancy_more_money_l39_3900


namespace NUMINAMATH_GPT_peter_initial_erasers_l39_3924

theorem peter_initial_erasers (E : ℕ) (h : E + 3 = 11) : E = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_peter_initial_erasers_l39_3924


namespace NUMINAMATH_GPT_angle_of_rotation_l39_3916

-- Definitions for the given conditions
def radius_large := 9 -- cm
def radius_medium := 3 -- cm
def radius_small := 1 -- cm
def speed := 1 -- cm/s

-- Definition of the angles calculations
noncomputable def rotations_per_revolution (R1 R2 : ℝ) : ℝ := R1 / R2
noncomputable def total_rotations (R1 R2 R3 : ℝ) : ℝ := 
  let rotations_medium := rotations_per_revolution R1 R2
  let net_rotations_medium := rotations_medium - 1
  net_rotations_medium * rotations_per_revolution R2 R3 + 1

-- Assertion to prove
theorem angle_of_rotation : 
  total_rotations radius_large radius_medium radius_small * 360 = 2520 :=
by 
  simp [total_rotations, rotations_per_revolution]
  exact sorry -- proof placeholder

end NUMINAMATH_GPT_angle_of_rotation_l39_3916


namespace NUMINAMATH_GPT_plains_routes_count_l39_3943

-- Defining the total number of cities and the number of cities in each region
def total_cities : Nat := 100
def mountainous_cities : Nat := 30
def plains_cities : Nat := total_cities - mountainous_cities

-- Defining the number of routes established each year and over three years
def routes_per_year : Nat := 50
def total_routes : Nat := routes_per_year * 3

-- Defining the number of routes connecting pairs of mountainous cities
def mountainous_routes : Nat := 21

-- The statement to prove the number of routes connecting pairs of plains cities
theorem plains_routes_count :
  plains_cities = 70 →
  total_routes = 150 →
  mountainous_routes = 21 →
  3 * mountainous_cities - 2 * mountainous_routes = 48 →
  3 * plains_cities - 48 = 162 →
  81 = 81 := sorry

end NUMINAMATH_GPT_plains_routes_count_l39_3943


namespace NUMINAMATH_GPT_total_weight_of_full_bucket_l39_3906

variable (a b x y : ℝ)

def bucket_weights :=
  (x + (1/3) * y = a) → (x + (3/4) * y = b) → (x + y = (16/5) * b - (11/5) * a)

theorem total_weight_of_full_bucket :
  bucket_weights a b x y :=
by
  intro h1 h2
  -- proof goes here, can be omitted as per instructions
  sorry

end NUMINAMATH_GPT_total_weight_of_full_bucket_l39_3906


namespace NUMINAMATH_GPT_general_term_of_arithmetic_seq_sum_of_first_n_terms_b_n_l39_3946

theorem general_term_of_arithmetic_seq
  (a_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (a_2_eq_3 : a_n 2 = 3)
  (S_4_eq_16 : S_n 4 = 16) :
  (∀ n, a_n n = 2 * n - 1) :=
sorry

theorem sum_of_first_n_terms_b_n
  (a_n : ℕ → ℕ)
  (b_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (general_formula_a_n : ∀ n, a_n n = 2 * n - 1)
  (b_n_definition : ∀ n, b_n n = 1 / (a_n n * a_n (n + 1))) :
  (∀ n, T_n n = n / (2 * n + 1)) :=
sorry

end NUMINAMATH_GPT_general_term_of_arithmetic_seq_sum_of_first_n_terms_b_n_l39_3946


namespace NUMINAMATH_GPT_cages_used_l39_3913

-- Define the initial conditions
def total_puppies : ℕ := 18
def puppies_sold : ℕ := 3
def puppies_per_cage : ℕ := 5

-- State the theorem to prove the number of cages used
theorem cages_used : (total_puppies - puppies_sold) / puppies_per_cage = 3 := by
  sorry

end NUMINAMATH_GPT_cages_used_l39_3913


namespace NUMINAMATH_GPT_chef_michel_total_pies_l39_3920

theorem chef_michel_total_pies
  (shepherd_pie_pieces : ℕ)
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ)
  (chicken_pot_pie_customers : ℕ)
  (H1 : shepherd_pie_pieces = 4)
  (H2 : chicken_pot_pie_pieces = 5)
  (H3 : shepherd_pie_customers = 52)
  (H4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) + (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by
  sorry

end NUMINAMATH_GPT_chef_michel_total_pies_l39_3920


namespace NUMINAMATH_GPT_total_money_correct_l39_3911

def total_money_in_cents : ℕ :=
  let Cindy := 5 * 10 + 3 * 50
  let Eric := 3 * 25 + 2 * 100 + 1 * 50
  let Garrick := 8 * 5 + 7 * 1
  let Ivy := 60 * 1 + 5 * 25
  let TotalBeforeRemoval := Cindy + Eric + Garrick + Ivy
  let BeaumontRemoval := 2 * 10 + 3 * 5 + 10 * 1
  let EricRemoval := 1 * 25 + 1 * 50
  TotalBeforeRemoval - BeaumontRemoval - EricRemoval

theorem total_money_correct : total_money_in_cents = 637 := by
  sorry

end NUMINAMATH_GPT_total_money_correct_l39_3911


namespace NUMINAMATH_GPT_distance_covered_at_40kmph_l39_3921

def total_distance : ℝ := 250
def speed_40 : ℝ := 40
def speed_60 : ℝ := 60
def total_time : ℝ := 5.2

theorem distance_covered_at_40kmph :
  ∃ (x : ℝ), (x / speed_40 + (total_distance - x) / speed_60 = total_time) ∧ x = 124 :=
  sorry

end NUMINAMATH_GPT_distance_covered_at_40kmph_l39_3921


namespace NUMINAMATH_GPT_complement_of_M_is_correct_l39_3941

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}
def complement_M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem complement_of_M_is_correct :
  (U \ M) = complement_M :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_is_correct_l39_3941


namespace NUMINAMATH_GPT_max_candies_ben_eats_l39_3908

theorem max_candies_ben_eats (total_candies : ℕ) (k : ℕ) (h_pos_k : k > 0) (b : ℕ) 
  (h_total : b + 2 * b + k * b = total_candies) (h_total_candies : total_candies = 30) : b = 6 :=
by
  -- placeholder for proof steps
  sorry

end NUMINAMATH_GPT_max_candies_ben_eats_l39_3908


namespace NUMINAMATH_GPT_sin_right_triangle_l39_3988

theorem sin_right_triangle (FG GH : ℝ) (h1 : FG = 13) (h2 : GH = 12) (h3 : FG^2 = FH^2 + GH^2) : 
  sin_H = 5 / 13 :=
by sorry

end NUMINAMATH_GPT_sin_right_triangle_l39_3988


namespace NUMINAMATH_GPT_least_times_to_eat_l39_3987

theorem least_times_to_eat (A B C : ℕ) (h1 : A = (9 * B) / 5) (h2 : B = C / 8) : 
  A = 2 ∧ B = 1 ∧ C = 8 :=
sorry

end NUMINAMATH_GPT_least_times_to_eat_l39_3987


namespace NUMINAMATH_GPT_find_a5_of_geom_seq_l39_3991

theorem find_a5_of_geom_seq 
  (a : ℕ → ℝ) (q : ℝ)
  (hgeom : ∀ n, a (n + 1) = a n * q)
  (S : ℕ → ℝ)
  (hS3 : S 3 = a 0 * (1 - q ^ 3) / (1 - q))
  (hS6 : S 6 = a 0 * (1 - q ^ 6) / (1 - q))
  (hS9 : S 9 = a 0 * (1 - q ^ 9) / (1 - q))
  (harith : S 3 + S 6 = 2 * S 9)
  (a8 : a 8 = 3) :
  a 5 = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_a5_of_geom_seq_l39_3991


namespace NUMINAMATH_GPT_rectangular_field_length_l39_3919

   theorem rectangular_field_length (w l : ℝ) 
     (h1 : l = 2 * w)
     (h2 : 64 = 8 * 8)
     (h3 : 64 = (1/72) * (l * w)) :
     l = 96 :=
   sorry
   
end NUMINAMATH_GPT_rectangular_field_length_l39_3919


namespace NUMINAMATH_GPT_math_problem_l39_3975

variable {a b : ℕ → ℕ}

-- Condition 1: a_n is an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n + a m

-- Condition 2: 2a₂ - a₇² + 2a₁₂ = 0
def satisfies_equation (a : ℕ → ℕ) : Prop :=
  2 * a 2 - (a 7)^2 + 2 * a 12 = 0

-- Condition 3: b_n is a geometric sequence
def is_geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, b (n + m) = b n * b m

-- Condition 4: b₇ = a₇
def b7_eq_a7 (a b : ℕ → ℕ) : Prop :=
  b 7 = a 7

-- To prove: b₅ * b₉ = 16
theorem math_problem (a b : ℕ → ℕ)
  (h₁ : is_arithmetic_sequence a)
  (h₂ : satisfies_equation a)
  (h₃ : is_geometric_sequence b)
  (h₄ : b7_eq_a7 a b) :
  b 5 * b 9 = 16 :=
sorry

end NUMINAMATH_GPT_math_problem_l39_3975


namespace NUMINAMATH_GPT_num_triangles_in_n_gon_l39_3951

-- Definitions for the problem in Lean based on provided conditions
def n_gon (n : ℕ) : Type := sorry  -- Define n-gon as a polygon with n sides
def non_intersecting_diagonals (n : ℕ) : Prop := sorry  -- Define the property of non-intersecting diagonals in an n-gon
def num_triangles (n : ℕ) : ℕ := sorry  -- Define a function to calculate the number of triangles formed by the diagonals in an n-gon

-- Statement of the theorem to prove
theorem num_triangles_in_n_gon (n : ℕ) (h : non_intersecting_diagonals n) : num_triangles n = n - 2 :=
by
  sorry

end NUMINAMATH_GPT_num_triangles_in_n_gon_l39_3951


namespace NUMINAMATH_GPT_probability_two_digit_between_21_and_30_l39_3960

theorem probability_two_digit_between_21_and_30 (dice1 dice2 : ℤ) (h1 : 1 ≤ dice1 ∧ dice1 ≤ 6) (h2 : 1 ≤ dice2 ∧ dice2 ≤ 6) :
∃ (p : ℚ), p = 11 / 36 := 
sorry

end NUMINAMATH_GPT_probability_two_digit_between_21_and_30_l39_3960


namespace NUMINAMATH_GPT_cone_volume_difference_l39_3997

theorem cone_volume_difference (H R : ℝ) : ΔV = (1/12) * Real.pi * R^2 * H := 
sorry

end NUMINAMATH_GPT_cone_volume_difference_l39_3997


namespace NUMINAMATH_GPT_points_per_enemy_l39_3972

theorem points_per_enemy (total_enemies : ℕ) (destroyed_enemies : ℕ) (total_points : ℕ) 
  (h1 : total_enemies = 7)
  (h2 : destroyed_enemies = total_enemies - 2)
  (h3 : destroyed_enemies = 5)
  (h4 : total_points = 40) :
  total_points / destroyed_enemies = 8 :=
by
  sorry

end NUMINAMATH_GPT_points_per_enemy_l39_3972


namespace NUMINAMATH_GPT_fraction_of_oil_sent_to_production_l39_3962

-- Definitions based on the problem's conditions
def initial_concentration : ℝ := 0.02
def replacement_concentration1 : ℝ := 0.03
def replacement_concentration2 : ℝ := 0.015
def final_concentration : ℝ := 0.02

-- Main theorem stating the fraction x is 1/2
theorem fraction_of_oil_sent_to_production (x : ℝ) (hx : x > 0) :
  (initial_concentration + (replacement_concentration1 - initial_concentration) * x) * (1 - x) +
  replacement_concentration2 * x = final_concentration →
  x = 0.5 :=
  sorry

end NUMINAMATH_GPT_fraction_of_oil_sent_to_production_l39_3962


namespace NUMINAMATH_GPT_fraction_of_students_with_buddy_l39_3929

theorem fraction_of_students_with_buddy (s n : ℕ) (h : n = 4 * s / 3) : 
  (n / 4 + s / 3) / (n + s : ℚ) = 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_students_with_buddy_l39_3929


namespace NUMINAMATH_GPT_correct_calculation_l39_3995

variable (a b : ℚ)

theorem correct_calculation :
  (a / b) ^ 4 = a ^ 4 / b ^ 4 := 
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l39_3995


namespace NUMINAMATH_GPT_find_value_of_N_l39_3914

theorem find_value_of_N (x N : ℝ) (hx : x = 1.5) (h : (3 + 2 * x)^5 = (N + 3 * x)^4) : N = 1.5 := by
  -- Here we will assume that the proof is filled in and correct.
  sorry

end NUMINAMATH_GPT_find_value_of_N_l39_3914


namespace NUMINAMATH_GPT_price_of_each_rose_l39_3970

def number_of_roses_started (roses : ℕ) : Prop := roses = 9
def number_of_roses_left (roses : ℕ) : Prop := roses = 4
def amount_earned (money : ℕ) : Prop := money = 35
def selling_price_per_rose (price : ℕ) : Prop := price = 7

theorem price_of_each_rose 
  (initial_roses sold_roses left_roses total_money price_per_rose : ℕ)
  (h1 : number_of_roses_started initial_roses)
  (h2 : number_of_roses_left left_roses)
  (h3 : amount_earned total_money)
  (h4 : initial_roses - left_roses = sold_roses)
  (h5 : total_money / sold_roses = price_per_rose) :
  selling_price_per_rose price_per_rose := 
by
  sorry

end NUMINAMATH_GPT_price_of_each_rose_l39_3970


namespace NUMINAMATH_GPT_find_circle_center_l39_3958

theorem find_circle_center : ∃ (h k : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 2*x + 12*y + 1 = 0 ↔ (x - h)^2 + (y - k)^2 = 36) ∧ h = 1 ∧ k = -6 := 
sorry

end NUMINAMATH_GPT_find_circle_center_l39_3958


namespace NUMINAMATH_GPT_max_ab_under_constraint_l39_3917

theorem max_ab_under_constraint (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 3 * a + 2 * b = 1) : 
  ab ≤ (1 / 24) ∧ (ab = 1 / 24 ↔ a = 1 / 6 ∧ b = 1 / 4) :=
sorry

end NUMINAMATH_GPT_max_ab_under_constraint_l39_3917


namespace NUMINAMATH_GPT_smallest_four_digit_mod_8_l39_3996

theorem smallest_four_digit_mod_8 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 8 = 5 ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ m % 8 = 5 → n ≤ m) → n = 1005 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_mod_8_l39_3996


namespace NUMINAMATH_GPT_binom_even_if_power_of_two_binom_odd_if_not_power_of_two_l39_3956

-- Definition of power of two
def is_power_of_two (n : ℕ) := ∃ m : ℕ, n = 2^m

-- Theorems to be proven
theorem binom_even_if_power_of_two (n : ℕ) (h : is_power_of_two n) :
  ∀ k : ℕ, 1 ≤ k ∧ k < n → Nat.choose n k % 2 = 0 := sorry

theorem binom_odd_if_not_power_of_two (n : ℕ) (h : ¬ is_power_of_two n) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ Nat.choose n k % 2 = 1 := sorry

end NUMINAMATH_GPT_binom_even_if_power_of_two_binom_odd_if_not_power_of_two_l39_3956


namespace NUMINAMATH_GPT_decimal_equivalent_of_fraction_squared_l39_3994

theorem decimal_equivalent_of_fraction_squared : (1 / 4 : ℝ) ^ 2 = 0.0625 :=
by sorry

end NUMINAMATH_GPT_decimal_equivalent_of_fraction_squared_l39_3994


namespace NUMINAMATH_GPT_bob_pennies_l39_3936

theorem bob_pennies (a b : ℤ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  sorry

end NUMINAMATH_GPT_bob_pennies_l39_3936


namespace NUMINAMATH_GPT_compute_B_93_l39_3902

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 0, 0], ![0, 0, -1], ![0, 1, 0]]

theorem compute_B_93 : B^93 = B := by
  sorry

end NUMINAMATH_GPT_compute_B_93_l39_3902


namespace NUMINAMATH_GPT_y_intercept_of_line_l39_3937

theorem y_intercept_of_line (m x y b : ℝ) (h1 : m = 4) (h2 : x = 50) (h3 : y = 300) (h4 : y = m * x + b) : b = 100 := by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l39_3937


namespace NUMINAMATH_GPT_expand_polynomial_l39_3903

theorem expand_polynomial :
  (3 * x^2 + 2 * x + 1) * (2 * x^2 + 3 * x + 4) = 6 * x^4 + 13 * x^3 + 20 * x^2 + 11 * x + 4 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l39_3903


namespace NUMINAMATH_GPT_sum_and_num_of_factors_eq_1767_l39_3948

theorem sum_and_num_of_factors_eq_1767 (n : ℕ) (σ d : ℕ → ℕ) :
  (σ n + d n = 1767) → 
  ∃ m : ℕ, σ m + d m = 1767 :=
by 
  sorry

end NUMINAMATH_GPT_sum_and_num_of_factors_eq_1767_l39_3948


namespace NUMINAMATH_GPT_abs_val_eq_two_l39_3944

theorem abs_val_eq_two (x : ℝ) (h : |x| = 2) : x = 2 ∨ x = -2 := 
sorry

end NUMINAMATH_GPT_abs_val_eq_two_l39_3944


namespace NUMINAMATH_GPT_binom_18_6_eq_18564_l39_3945

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binom_18_6_eq_18564 : binomial 18 6 = 18564 := by
  sorry

end NUMINAMATH_GPT_binom_18_6_eq_18564_l39_3945


namespace NUMINAMATH_GPT_cubic_sum_l39_3971

theorem cubic_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 14) : x ^ 3 + y ^ 3 = 580 :=
by 
  sorry

end NUMINAMATH_GPT_cubic_sum_l39_3971


namespace NUMINAMATH_GPT_range_of_x_l39_3931

theorem range_of_x (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) : x^2 + a * x > 4 * x + a - 3 ↔ (x > 3 ∨ x < -1) := by
  sorry

end NUMINAMATH_GPT_range_of_x_l39_3931


namespace NUMINAMATH_GPT_time_for_Q_l39_3964

-- Definitions of conditions
def time_for_P := 252
def meet_time := 2772

-- Main statement to prove
theorem time_for_Q : (∃ T : ℕ, lcm time_for_P T = meet_time) ∧ (lcm time_for_P meet_time = meet_time) :=
    by 
    sorry

end NUMINAMATH_GPT_time_for_Q_l39_3964


namespace NUMINAMATH_GPT_bejgli_slices_l39_3954

theorem bejgli_slices (x : ℕ) (hx : x ≤ 58) 
    (h1 : x * (x - 1) * (x - 2) = 3 * (58 - x) * (57 - x) * x) : 
    58 - x = 21 :=
by
  have hpos1 : 0 < x := sorry  -- x should be strictly positive since it's a count
  have hpos2 : 0 < 58 - x := sorry  -- the remaining slices should be strictly positive
  sorry

end NUMINAMATH_GPT_bejgli_slices_l39_3954


namespace NUMINAMATH_GPT_decrease_in_radius_l39_3966

theorem decrease_in_radius
  (dist_summer : ℝ)
  (dist_winter : ℝ)
  (radius_summer : ℝ) 
  (mile_to_inch : ℝ)
  (π : ℝ) 
  (δr : ℝ) :
  dist_summer = 560 →
  dist_winter = 570 →
  radius_summer = 20 →
  mile_to_inch = 63360 →
  π = Real.pi →
  δr = 0.33 :=
sorry

end NUMINAMATH_GPT_decrease_in_radius_l39_3966


namespace NUMINAMATH_GPT_min_value_expression_l39_3930

/-- 
Given real numbers a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p such that 
abcd = 16, efgh = 16, ijkl = 16, and mnop = 16, prove that the minimum value of 
(aeim)^2 + (bfjn)^2 + (cgko)^2 + (dhlp)^2 is 1024. 
-/
theorem min_value_expression (a b c d e f g h i j k l m n o p : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 16) 
  (h3 : i * j * k * l = 16) 
  (h4 : m * n * o * p = 16) : 
  (a * e * i * m) ^ 2 + (b * f * j * n) ^ 2 + (c * g * k * o) ^ 2 + (d * h * l * p) ^ 2 ≥ 1024 :=
by 
  sorry


end NUMINAMATH_GPT_min_value_expression_l39_3930
