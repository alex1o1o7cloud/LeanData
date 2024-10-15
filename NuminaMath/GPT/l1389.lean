import Mathlib

namespace NUMINAMATH_GPT_pentagon_angle_sum_l1389_138970

theorem pentagon_angle_sum
  (a b c d : ℝ) (Q : ℝ)
  (sum_angles : 180 * (5 - 2) = 540)
  (given_angles : a = 130 ∧ b = 80 ∧ c = 105 ∧ d = 110) :
  Q = 540 - (a + b + c + d) := by
  sorry

end NUMINAMATH_GPT_pentagon_angle_sum_l1389_138970


namespace NUMINAMATH_GPT_calculate_total_cost_l1389_138928

-- Define the cost per workbook
def cost_per_workbook (x : ℝ) : ℝ := x

-- Define the number of workbooks
def number_of_workbooks : ℝ := 400

-- Define the total cost calculation
def total_cost (x : ℝ) : ℝ := number_of_workbooks * cost_per_workbook x

-- State the theorem to prove
theorem calculate_total_cost (x : ℝ) : total_cost x = 400 * x :=
by sorry

end NUMINAMATH_GPT_calculate_total_cost_l1389_138928


namespace NUMINAMATH_GPT_crystal_barrette_sets_l1389_138971

-- Definitional and situational context
def cost_of_barrette : ℕ := 3
def cost_of_comb : ℕ := 1
def kristine_total_cost : ℕ := 4
def total_spent : ℕ := 14

-- The Lean 4 theorem statement to prove that Crystal bought 3 sets of barrettes
theorem crystal_barrette_sets (x : ℕ) 
  (kristine_cost : kristine_total_cost = cost_of_barrette + cost_of_comb + 1)
  (total_cost_eq : kristine_total_cost + (x * cost_of_barrette + cost_of_comb) = total_spent) 
  : x = 3 := 
sorry

end NUMINAMATH_GPT_crystal_barrette_sets_l1389_138971


namespace NUMINAMATH_GPT_probability_one_hits_correct_l1389_138991

-- Define the probabilities for A hitting and B hitting
noncomputable def P_A : ℝ := 0.4
noncomputable def P_B : ℝ := 0.5

-- Calculate the required probability
noncomputable def probability_one_hits : ℝ :=
  P_A * (1 - P_B) + (1 - P_A) * P_B

-- Statement of the theorem
theorem probability_one_hits_correct :
  probability_one_hits = 0.5 := by 
  sorry

end NUMINAMATH_GPT_probability_one_hits_correct_l1389_138991


namespace NUMINAMATH_GPT_remaining_candy_l1389_138916

def initial_candy : ℕ := 36
def ate_candy1 : ℕ := 17
def ate_candy2 : ℕ := 15
def total_ate_candy : ℕ := ate_candy1 + ate_candy2

theorem remaining_candy : initial_candy - total_ate_candy = 4 := by
  sorry

end NUMINAMATH_GPT_remaining_candy_l1389_138916


namespace NUMINAMATH_GPT_points_on_same_circle_l1389_138920
open Real

theorem points_on_same_circle (m : ℝ) :
  ∃ D E F, 
  (2^2 + 1^2 + 2 * D + 1 * E + F = 0) ∧
  (4^2 + 2^2 + 4 * D + 2 * E + F = 0) ∧
  (3^2 + 4^2 + 3 * D + 4 * E + F = 0) ∧
  (1^2 + m^2 + 1 * D + m * E + F = 0) →
  (m = 2 ∨ m = 3) := 
sorry

end NUMINAMATH_GPT_points_on_same_circle_l1389_138920


namespace NUMINAMATH_GPT_simplify_expression_l1389_138909

variable (y : ℝ)
variable (h : y ≠ 0)

theorem simplify_expression : (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1389_138909


namespace NUMINAMATH_GPT_find_m_from_parallel_vectors_l1389_138934

variables (m : ℝ)

def a : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (2, -3)

-- The condition that vectors a and b are parallel
def vectors_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u.1 = k * v.1 ∧ u.2 = k * v.2

-- Given that a and b are parallel, prove that m = -3/2
theorem find_m_from_parallel_vectors
  (h : vectors_parallel (1, m) (2, -3)) :
  m = -3 / 2 :=
sorry

end NUMINAMATH_GPT_find_m_from_parallel_vectors_l1389_138934


namespace NUMINAMATH_GPT_sum_of_first_four_terms_l1389_138907

noncomputable def sum_first_n_terms (a q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem sum_of_first_four_terms :
  ∀ (a q : ℝ), a * (1 + q) = 7 → a * (q^6 - 1) / (q - 1) = 91 →
  a * (1 + q + q^2 + q^3) = 28 :=
by
  intros a q h₁ h₂
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sum_of_first_four_terms_l1389_138907


namespace NUMINAMATH_GPT_oil_bill_increase_l1389_138964

theorem oil_bill_increase :
  ∀ (F x : ℝ), 
    (F / 120 = 5 / 4) → 
    ((F + x) / 120 = 3 / 2) → 
    x = 30 :=
by
  intros F x h1 h2
  -- proof
  sorry

end NUMINAMATH_GPT_oil_bill_increase_l1389_138964


namespace NUMINAMATH_GPT_average_percentage_15_students_l1389_138951

-- Define the average percentage of the 15 students
variable (x : ℝ)

-- Condition 1: Total percentage for the 15 students is 15 * x
def total_15_students : ℝ := 15 * x

-- Condition 2: Total percentage for the 10 students who averaged 88%
def total_10_students : ℝ := 10 * 88

-- Condition 3: Total percentage for all 25 students who averaged 79%
def total_all_students : ℝ := 25 * 79

-- Mathematical problem: Prove that x = 73 given the conditions.
theorem average_percentage_15_students (h : total_15_students x + total_10_students = total_all_students) : x = 73 := 
by
  sorry

end NUMINAMATH_GPT_average_percentage_15_students_l1389_138951


namespace NUMINAMATH_GPT_assign_roles_l1389_138939

def maleRoles : ℕ := 3
def femaleRoles : ℕ := 3
def eitherGenderRoles : ℕ := 4
def menCount : ℕ := 7
def womenCount : ℕ := 8

theorem assign_roles : 
  (menCount.choose maleRoles) * 
  (womenCount.choose femaleRoles) * 
  ((menCount + womenCount - maleRoles - femaleRoles).choose eitherGenderRoles) = 213955200 := 
  sorry

end NUMINAMATH_GPT_assign_roles_l1389_138939


namespace NUMINAMATH_GPT_quadratic_to_square_form_l1389_138952

theorem quadratic_to_square_form (x m n : ℝ) (h : x^2 + 6 * x - 1 = 0) 
  (hm : m = 3) (hn : n = 10) : m - n = -7 :=
by 
  -- Proof steps (skipped, as per instructions)
  sorry

end NUMINAMATH_GPT_quadratic_to_square_form_l1389_138952


namespace NUMINAMATH_GPT_compare_two_sqrt_three_l1389_138940

theorem compare_two_sqrt_three : 2 > Real.sqrt 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_compare_two_sqrt_three_l1389_138940


namespace NUMINAMATH_GPT_Natasha_avg_speed_climb_l1389_138902

-- Definitions for conditions
def distance_to_top : ℝ := sorry -- We need to find this
def time_up := 3 -- time in hours to climb up
def time_down := 2 -- time in hours to climb down
def avg_speed_journey := 3 -- avg speed in km/hr for the whole journey

-- Equivalent math proof problem statement
theorem Natasha_avg_speed_climb (distance_to_top : ℝ) 
  (h1 : time_up = 3)
  (h2 : time_down = 2)
  (h3 : avg_speed_journey = 3)
  (h4 : (2 * distance_to_top) / (time_up + time_down) = avg_speed_journey) : 
  (distance_to_top / time_up) = 2.5 :=
sorry -- Proof not required

end NUMINAMATH_GPT_Natasha_avg_speed_climb_l1389_138902


namespace NUMINAMATH_GPT_find_interval_l1389_138953

theorem find_interval (x : ℝ) : x^2 + 3*x < 10 ↔ -5 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_find_interval_l1389_138953


namespace NUMINAMATH_GPT_elly_candies_l1389_138908

theorem elly_candies (a b c : ℝ) (h1 : a * b * c = 216) : 
  24 * 216 = 5184 :=
by
  sorry

end NUMINAMATH_GPT_elly_candies_l1389_138908


namespace NUMINAMATH_GPT_second_discount_percentage_l1389_138982

theorem second_discount_percentage (x : ℝ) :
  9356.725146198829 * 0.8 * (1 - x / 100) * 0.95 = 6400 → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_second_discount_percentage_l1389_138982


namespace NUMINAMATH_GPT_plane_ratio_l1389_138967

section

variables (D B T P : ℕ)

-- Given conditions
axiom total_distance : D = 1800
axiom distance_by_bus : B = 720
axiom distance_by_train : T = (2 * B) / 3

-- Prove the ratio of the distance traveled by plane to the whole trip
theorem plane_ratio :
  D = 1800 →
  B = 720 →
  T = (2 * B) / 3 →
  P = D - (T + B) →
  P / D = 1 / 3 := by
  intros h1 h2 h3 h4
  sorry

end

end NUMINAMATH_GPT_plane_ratio_l1389_138967


namespace NUMINAMATH_GPT_probability_not_paired_shoes_l1389_138905

noncomputable def probability_not_pair (total_shoes : ℕ) (pairs : ℕ) (shoes_drawn : ℕ) : ℚ :=
  let total_ways := Nat.choose total_shoes shoes_drawn
  let pair_ways := pairs * Nat.choose 2 2
  let not_pair_ways := total_ways - pair_ways
  not_pair_ways / total_ways

theorem probability_not_paired_shoes (total_shoes pairs shoes_drawn : ℕ) (h1 : total_shoes = 6) 
(h2 : pairs = 3) (h3 : shoes_drawn = 2) :
  probability_not_pair total_shoes pairs shoes_drawn = 4 / 5 :=
by 
  rw [h1, h2, h3]
  simp [probability_not_pair, Nat.choose]
  sorry

end NUMINAMATH_GPT_probability_not_paired_shoes_l1389_138905


namespace NUMINAMATH_GPT_polynomial_discriminant_l1389_138925

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_discriminant_l1389_138925


namespace NUMINAMATH_GPT_roberto_outfits_l1389_138995

theorem roberto_outfits : 
  let trousers := 5
  let shirts := 5
  let jackets := 3
  (trousers * shirts * jackets = 75) :=
by sorry

end NUMINAMATH_GPT_roberto_outfits_l1389_138995


namespace NUMINAMATH_GPT_average_speed_is_correct_l1389_138912
noncomputable def average_speed_trip : ℝ :=
  let distance_AB := 240 * 5
  let distance_BC := 300 * 3
  let distance_CD := 400 * 4
  let total_distance := distance_AB + distance_BC + distance_CD
  let flight_time_AB := 5
  let layover_B := 2
  let flight_time_BC := 3
  let layover_C := 1
  let flight_time_CD := 4
  let total_time := (flight_time_AB + flight_time_BC + flight_time_CD) + (layover_B + layover_C)
  total_distance / total_time

theorem average_speed_is_correct :
  average_speed_trip = 246.67 := sorry

end NUMINAMATH_GPT_average_speed_is_correct_l1389_138912


namespace NUMINAMATH_GPT_area_enclosed_by_sin_l1389_138985

/-- The area of the figure enclosed by the curve y = sin(x), the lines x = -π/3, x = π/2, and the x-axis is 3/2. -/
theorem area_enclosed_by_sin (x y : ℝ) (h : y = Real.sin x) (a b : ℝ) 
(h1 : a = -Real.pi / 3) (h2 : b = Real.pi / 2) :
  ∫ x in a..b, |Real.sin x| = 3 / 2 := 
sorry

end NUMINAMATH_GPT_area_enclosed_by_sin_l1389_138985


namespace NUMINAMATH_GPT_distance_traveled_on_fifth_day_equals_12_li_l1389_138957

theorem distance_traveled_on_fifth_day_equals_12_li:
  ∀ {a_1 : ℝ},
    (a_1 * ((1 - (1 / 2) ^ 6) / (1 - 1 / 2)) = 378) →
    (a_1 * (1 / 2) ^ 4 = 12) :=
by
  intros a_1 h
  sorry

end NUMINAMATH_GPT_distance_traveled_on_fifth_day_equals_12_li_l1389_138957


namespace NUMINAMATH_GPT_sum_of_integers_is_28_l1389_138980

theorem sum_of_integers_is_28 (m n p q : ℕ) (hmnpq_diff : m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q)
  (hm_pos : 0 < m) (hn_pos : 0 < n) (hp_pos : 0 < p) (hq_pos : 0 < q)
  (h_prod : (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4) : m + n + p + q = 28 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_is_28_l1389_138980


namespace NUMINAMATH_GPT_integer_solution_n_l1389_138933

theorem integer_solution_n 
  (n : Int) 
  (h1 : n + 13 > 15) 
  (h2 : -6 * n > -18) : 
  n = 2 := 
sorry

end NUMINAMATH_GPT_integer_solution_n_l1389_138933


namespace NUMINAMATH_GPT_shooter_random_event_l1389_138993

def eventA := "The sun rises from the east"
def eventB := "A coin thrown up from the ground will fall down"
def eventC := "A shooter hits the target with 10 points in one shot"
def eventD := "Xiao Ming runs at a speed of 30 meters per second"

def is_random_event (event : String) := event = eventC

theorem shooter_random_event : is_random_event eventC := 
by
  sorry

end NUMINAMATH_GPT_shooter_random_event_l1389_138993


namespace NUMINAMATH_GPT_dan_licks_l1389_138973

/-- 
Given that Michael takes 63 licks, Sam takes 70 licks, David takes 70 licks, 
Lance takes 39 licks, and the average number of licks for all five people is 60, 
prove that Dan takes 58 licks to get to the center of a lollipop.
-/
theorem dan_licks (D : ℕ) 
  (M : ℕ := 63) 
  (S : ℕ := 70) 
  (Da : ℕ := 70) 
  (L : ℕ := 39)
  (avg : ℕ := 60) :
  ((M + S + Da + L + D) / 5 = avg) → D = 58 :=
by sorry

end NUMINAMATH_GPT_dan_licks_l1389_138973


namespace NUMINAMATH_GPT_sara_walking_distance_l1389_138979

noncomputable def circle_area := 616
noncomputable def pi_estimate := (22: ℚ) / 7
noncomputable def extra_distance := 3

theorem sara_walking_distance (r : ℚ) (radius_pos : 0 < r) : 
  pi_estimate * r^2 = circle_area →
  2 * pi_estimate * r + extra_distance = 91 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sara_walking_distance_l1389_138979


namespace NUMINAMATH_GPT_base_conversion_b_l1389_138984

-- Define the problem in Lean
theorem base_conversion_b (b : ℕ) : 
  (b^2 + 2 * b - 16 = 0) → b = 4 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_base_conversion_b_l1389_138984


namespace NUMINAMATH_GPT_incorrect_statement_l1389_138946

theorem incorrect_statement (a : ℝ) (x : ℝ) (h : a > 1) :
  ¬((x = 0 → a^x = 1) ∧
    (x = 1 → a^x = a) ∧
    (x = -1 → a^x = 1/a) ∧
    (x < 0 → 0 < a^x ∧ ∀ ε > 0, ∃ x' < x, a^x' < ε)) :=
sorry

end NUMINAMATH_GPT_incorrect_statement_l1389_138946


namespace NUMINAMATH_GPT_inequality_proof_l1389_138997

noncomputable def f (a x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log (1 + x) - x

theorem inequality_proof (a b : ℝ) (ha : 1 < a) (hb : 0 < b) : 
  f a (a + b) > f a 1 → g (a / b) < g 0 → 1 / (a + b) < Real.log (a + b) / b ∧ Real.log (a + b) / b < a / b := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1389_138997


namespace NUMINAMATH_GPT_P_lt_Q_l1389_138943

noncomputable def P (a : ℝ) : ℝ := (Real.sqrt (a + 41)) - (Real.sqrt (a + 40))
noncomputable def Q (a : ℝ) : ℝ := (Real.sqrt (a + 39)) - (Real.sqrt (a + 38))

theorem P_lt_Q (a : ℝ) (h : a > -38) : P a < Q a := by sorry

end NUMINAMATH_GPT_P_lt_Q_l1389_138943


namespace NUMINAMATH_GPT_least_distance_between_ticks_l1389_138927

theorem least_distance_between_ticks :
  ∃ z : ℝ, ∀ (a b : ℤ), (a / 5 ≠ b / 7) → abs (a / 5 - b / 7) = (1 / 35) := 
sorry

end NUMINAMATH_GPT_least_distance_between_ticks_l1389_138927


namespace NUMINAMATH_GPT_product_increase_by_13_exists_l1389_138938

theorem product_increase_by_13_exists :
  ∃ a1 a2 a3 a4 a5 a6 a7 : ℕ,
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * (a1 * a2 * a3 * a4 * a5 * a6 * a7)) :=
by
  sorry

end NUMINAMATH_GPT_product_increase_by_13_exists_l1389_138938


namespace NUMINAMATH_GPT_min_alpha_beta_l1389_138942

theorem min_alpha_beta (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1)
  (alpha : ℝ := a + 1 / a) (beta : ℝ := b + 1 / b) :
  alpha + beta ≥ 10 := by
  sorry

end NUMINAMATH_GPT_min_alpha_beta_l1389_138942


namespace NUMINAMATH_GPT_least_five_digit_congruent_to_8_mod_17_l1389_138906

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∃ n : ℕ, n = 10004 ∧ n % 17 = 8 ∧ 10000 ≤ n ∧ n < 100000 ∧ (∀ m, 10000 ≤ m ∧ m < 10004 → m % 17 ≠ 8) :=
sorry

end NUMINAMATH_GPT_least_five_digit_congruent_to_8_mod_17_l1389_138906


namespace NUMINAMATH_GPT_check_bag_correct_l1389_138949

-- Define the conditions as variables and statements
variables (uber_to_house : ℕ) (uber_to_airport : ℕ) (check_bag : ℕ)
          (security : ℕ) (wait_for_boarding : ℕ) (wait_for_takeoff : ℕ) (total_time : ℕ)

-- Assign the given conditions
def given_conditions : Prop :=
  uber_to_house = 10 ∧
  uber_to_airport = 5 * uber_to_house ∧
  security = 3 * check_bag ∧
  wait_for_boarding = 20 ∧
  wait_for_takeoff = 2 * wait_for_boarding ∧
  total_time = 180

-- Define the question as a statement
def check_bag_time (check_bag : ℕ) : Prop :=
  check_bag = 15

-- The Lean theorem based on the problem, conditions, and answer
theorem check_bag_correct :
  given_conditions uber_to_house uber_to_airport check_bag security wait_for_boarding wait_for_takeoff total_time →
  check_bag_time check_bag :=
by
  intros h
  sorry

end NUMINAMATH_GPT_check_bag_correct_l1389_138949


namespace NUMINAMATH_GPT_jane_weekly_pages_l1389_138961

-- Define the daily reading amounts
def monday_wednesday_morning_pages : ℕ := 5
def monday_wednesday_evening_pages : ℕ := 10
def tuesday_thursday_morning_pages : ℕ := 7
def tuesday_thursday_evening_pages : ℕ := 8
def friday_morning_pages : ℕ := 10
def friday_evening_pages : ℕ := 15
def weekend_morning_pages : ℕ := 12
def weekend_evening_pages : ℕ := 20

-- Define the number of days
def monday_wednesday_days : ℕ := 2
def tuesday_thursday_days : ℕ := 2
def friday_days : ℕ := 1
def weekend_days : ℕ := 2

-- Function to calculate weekly pages
def weekly_pages :=
  (monday_wednesday_days * (monday_wednesday_morning_pages + monday_wednesday_evening_pages)) +
  (tuesday_thursday_days * (tuesday_thursday_morning_pages + tuesday_thursday_evening_pages)) +
  (friday_days * (friday_morning_pages + friday_evening_pages)) +
  (weekend_days * (weekend_morning_pages + weekend_evening_pages))

-- Proof statement
theorem jane_weekly_pages : weekly_pages = 149 := by
  unfold weekly_pages
  norm_num
  sorry

end NUMINAMATH_GPT_jane_weekly_pages_l1389_138961


namespace NUMINAMATH_GPT_mary_saves_in_five_months_l1389_138968

def washing_earnings : ℕ := 20
def walking_earnings : ℕ := 40
def monthly_earnings : ℕ := washing_earnings + walking_earnings
def savings_rate : ℕ := 2
def monthly_savings : ℕ := monthly_earnings / savings_rate
def total_savings_target : ℕ := 150

theorem mary_saves_in_five_months :
  total_savings_target / monthly_savings = 5 :=
by
  sorry

end NUMINAMATH_GPT_mary_saves_in_five_months_l1389_138968


namespace NUMINAMATH_GPT_platform_length_proof_l1389_138996

-- Given conditions
def train_length : ℝ := 300
def time_to_cross_platform : ℝ := 27
def time_to_cross_pole : ℝ := 18

-- The length of the platform L to be proved
def length_of_platform (L : ℝ) : Prop := 
  (train_length / time_to_cross_pole) = (train_length + L) / time_to_cross_platform

theorem platform_length_proof : length_of_platform 150 :=
by
  sorry

end NUMINAMATH_GPT_platform_length_proof_l1389_138996


namespace NUMINAMATH_GPT_solve_equation_simplify_expression_l1389_138935

-- Part 1: Solving the equation
theorem solve_equation (x : ℝ) : 9 * (x - 3) ^ 2 - 121 = 0 ↔ x = 20 / 3 ∨ x = -2 / 3 :=
by 
    sorry

-- Part 2: Simplifying the expression
theorem simplify_expression (x y : ℝ) : (x - 2 * y) * (x ^ 2 + 2 * x * y + 4 * y ^ 2) = x ^ 3 - 8 * y ^ 3 :=
by 
    sorry

end NUMINAMATH_GPT_solve_equation_simplify_expression_l1389_138935


namespace NUMINAMATH_GPT_group_total_cost_l1389_138931

noncomputable def total_cost
  (num_people : Nat) 
  (cost_per_person : Nat) : Nat :=
  num_people * cost_per_person

theorem group_total_cost (num_people := 15) (cost_per_person := 900) :
  total_cost num_people cost_per_person = 13500 :=
by
  sorry

end NUMINAMATH_GPT_group_total_cost_l1389_138931


namespace NUMINAMATH_GPT_corrected_mean_is_124_931_l1389_138994

/-
Given:
- original_mean : Real = 125.6
- num_observations : Nat = 100
- incorrect_obs1 : Real = 95.3
- incorrect_obs2 : Real = -15.9
- correct_obs1 : Real = 48.2
- correct_obs2 : Real = -35.7

Prove:
- new_mean == 124.931
-/

noncomputable def original_mean : ℝ := 125.6
def num_observations : ℕ := 100
noncomputable def incorrect_obs1 : ℝ := 95.3
noncomputable def incorrect_obs2 : ℝ := -15.9
noncomputable def correct_obs1 : ℝ := 48.2
noncomputable def correct_obs2 : ℝ := -35.7

noncomputable def incorrect_total_sum : ℝ := original_mean * num_observations
noncomputable def sum_incorrect_obs : ℝ := incorrect_obs1 + incorrect_obs2
noncomputable def sum_correct_obs : ℝ := correct_obs1 + correct_obs2
noncomputable def corrected_total_sum : ℝ := incorrect_total_sum - sum_incorrect_obs + sum_correct_obs
noncomputable def new_mean : ℝ := corrected_total_sum / num_observations

theorem corrected_mean_is_124_931 : new_mean = 124.931 := sorry

end NUMINAMATH_GPT_corrected_mean_is_124_931_l1389_138994


namespace NUMINAMATH_GPT_yardage_lost_due_to_sacks_l1389_138976

theorem yardage_lost_due_to_sacks 
  (throws : ℕ)
  (percent_no_throw : ℝ)
  (half_sack_prob : ℕ)
  (sack_pattern : ℕ → ℕ)
  (correct_answer : ℕ) : 
  throws = 80 →
  percent_no_throw = 0.30 →
  (∀ (n: ℕ), half_sack_prob = n/2) →
  (sack_pattern 1 = 3 ∧ sack_pattern 2 = 5 ∧ ∀ n, n > 2 → sack_pattern n = sack_pattern (n - 1) + 2) →
  correct_answer = 168 :=
by
  sorry

end NUMINAMATH_GPT_yardage_lost_due_to_sacks_l1389_138976


namespace NUMINAMATH_GPT_sum_of_midpoint_coordinates_l1389_138972

theorem sum_of_midpoint_coordinates : 
  let (x1, y1) := (4, 7)
  let (x2, y2) := (10, 19)
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  sum_of_coordinates = 20 := sorry

end NUMINAMATH_GPT_sum_of_midpoint_coordinates_l1389_138972


namespace NUMINAMATH_GPT_cos_diff_proof_l1389_138922

theorem cos_diff_proof (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2)
  (h2 : Real.cos A + Real.cos B = 1) :
  Real.cos (A - B) = 5 / 8 := 
by
  sorry

end NUMINAMATH_GPT_cos_diff_proof_l1389_138922


namespace NUMINAMATH_GPT_no_two_right_angles_in_triangle_l1389_138989

theorem no_two_right_angles_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 90) (h3 : B = 90): false :=
by
  -- we assume A = 90 and B = 90,
  -- then A + B + C > 180, which contradicts h1,
  sorry
  
example : (3 = 3) := by sorry  -- Given the context of the multiple-choice problem.

end NUMINAMATH_GPT_no_two_right_angles_in_triangle_l1389_138989


namespace NUMINAMATH_GPT_find_n_l1389_138983

theorem find_n (x y m n : ℕ) (hx : 0 ≤ x ∧ x < 10) (hy : 0 ≤ y ∧ y < 10) 
  (h1 : 100 * y + x = (x + y) * m) (h2 : 100 * x + y = (x + y) * n) : n = 101 - m :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1389_138983


namespace NUMINAMATH_GPT_winston_cents_left_l1389_138999

-- Definitions based on the conditions in the problem
def quarters := 14
def cents_per_quarter := 25
def half_dollar_in_cents := 50

-- Formulation of the problem statement in Lean
theorem winston_cents_left : (quarters * cents_per_quarter) - half_dollar_in_cents = 300 :=
by sorry

end NUMINAMATH_GPT_winston_cents_left_l1389_138999


namespace NUMINAMATH_GPT_marks_change_factor_l1389_138918

def total_marks (n : ℕ) (avg : ℝ) : ℝ := n * avg

theorem marks_change_factor 
  (n : ℕ) (initial_avg new_avg : ℝ) 
  (initial_total := total_marks n initial_avg) 
  (new_total := total_marks n new_avg)
  (h1 : initial_avg = 36)
  (h2 : new_avg = 72)
  (h3 : n = 12):
  (new_total / initial_total) = 2 :=
by
  sorry

end NUMINAMATH_GPT_marks_change_factor_l1389_138918


namespace NUMINAMATH_GPT_fibonacci_series_sum_l1389_138977

-- Definition of the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Theorem to prove that the infinite series sum is 2
theorem fibonacci_series_sum : (∑' n : ℕ, (fib n : ℝ) / (2 ^ n : ℝ)) = 2 :=
sorry

end NUMINAMATH_GPT_fibonacci_series_sum_l1389_138977


namespace NUMINAMATH_GPT_toms_age_ratio_l1389_138958

variable (T N : ℕ)

def toms_age_condition : Prop :=
  T = 3 * (T - 4 * N) + N

theorem toms_age_ratio (h : toms_age_condition T N) : T / N = 11 / 2 :=
by sorry

end NUMINAMATH_GPT_toms_age_ratio_l1389_138958


namespace NUMINAMATH_GPT_roots_sum_of_squares_l1389_138921

theorem roots_sum_of_squares {p q r : ℝ} 
  (h₁ : ∀ x : ℝ, (x - p) * (x - q) * (x - r) = x^3 - 24 * x^2 + 50 * x - 35) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 1052 :=
by
  have h_sum : p + q + r = 24 := by sorry
  have h_product : p * q + q * r + r * p = 50 := by sorry
  sorry

end NUMINAMATH_GPT_roots_sum_of_squares_l1389_138921


namespace NUMINAMATH_GPT_rice_mixing_ratio_l1389_138956

theorem rice_mixing_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (4.5 * x + 8.75 * y) / (x + y) = 7.5 → y / x = 2.4 :=
by
  sorry

end NUMINAMATH_GPT_rice_mixing_ratio_l1389_138956


namespace NUMINAMATH_GPT_building_houses_200_people_l1389_138932

-- Define number of floors, apartments per floor, and people per apartment as constants
def numFloors := 25
def apartmentsPerFloor := 4
def peoplePerApartment := 2

-- Define the total number of apartments
def totalApartments := numFloors * apartmentsPerFloor

-- Define the total number of people
def totalPeople := totalApartments * peoplePerApartment

theorem building_houses_200_people : totalPeople = 200 :=
by
  sorry

end NUMINAMATH_GPT_building_houses_200_people_l1389_138932


namespace NUMINAMATH_GPT_notes_count_l1389_138959

theorem notes_count (x : ℕ) (num_2_yuan num_5_yuan num_10_yuan total_notes total_amount : ℕ) 
    (h1 : total_amount = 160)
    (h2 : total_notes = 25)
    (h3 : num_5_yuan = x)
    (h4 : num_10_yuan = x)
    (h5 : num_2_yuan = total_notes - 2 * x)
    (h6 : 2 * num_2_yuan + 5 * num_5_yuan + 10 * num_10_yuan = total_amount) :
    num_5_yuan = 10 ∧ num_10_yuan = 10 ∧ num_2_yuan = 5 :=
by
  sorry

end NUMINAMATH_GPT_notes_count_l1389_138959


namespace NUMINAMATH_GPT_oranges_in_bowl_l1389_138903

theorem oranges_in_bowl (bananas : Nat) (apples : Nat) (pears : Nat) (total_fruits : Nat) (h_bananas : bananas = 4) (h_apples : apples = 3 * bananas) (h_pears : pears = 5) (h_total_fruits : total_fruits = 30) :
  total_fruits - (bananas + apples + pears) = 9 :=
by
  subst h_bananas
  subst h_apples
  subst h_pears
  subst h_total_fruits
  sorry

end NUMINAMATH_GPT_oranges_in_bowl_l1389_138903


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l1389_138974

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  ∀ (c : ℝ), c = (1 / a) + (4 / b) → c ≥ 9 :=
by
  intros c hc
  sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l1389_138974


namespace NUMINAMATH_GPT_find_z_l1389_138936

variable (x y z : ℝ)

-- Define x, y as given in the problem statement
def x_def : x = (Real.sqrt 7 + Real.sqrt 3) / (Real.sqrt 7 - Real.sqrt 3) := by
  sorry

def y_def : y = (Real.sqrt 7 - Real.sqrt 3) / (Real.sqrt 7 + Real.sqrt 3) := by
  sorry

-- Define the equation relating z to x and y
def z_eq : 192 * z = x^4 + y^4 + (x + y)^4 := by 
  sorry

-- Theorem stating the value of z
theorem find_z (h1 : x = (Real.sqrt 7 + Real.sqrt 3) / (Real.sqrt 7 - Real.sqrt 3))
               (h2 : y = (Real.sqrt 7 - Real.sqrt 3) / (Real.sqrt 7 + Real.sqrt 3))
               (h3 : 192 * z = x^4 + y^4 + (x + y)^4) :
  z = 6 := by 
  sorry

end NUMINAMATH_GPT_find_z_l1389_138936


namespace NUMINAMATH_GPT_g_increasing_on_minus_infty_one_l1389_138930

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)
noncomputable def f_inv (x : ℝ) : ℝ := (x + 1) / (x - 1)
noncomputable def g (x : ℝ) : ℝ := 1 + (2 * x) / (1 - x)

theorem g_increasing_on_minus_infty_one : (∀ x y : ℝ, x < y → x < 1 → y ≤ 1 → g x < g y) :=
sorry

end NUMINAMATH_GPT_g_increasing_on_minus_infty_one_l1389_138930


namespace NUMINAMATH_GPT_describe_graph_l1389_138966

noncomputable def points_satisfying_equation (x y : ℝ) : Prop :=
  (x - y) ^ 2 = x ^ 2 + y ^ 2

theorem describe_graph : {p : ℝ × ℝ | points_satisfying_equation p.1 p.2} = {p : ℝ × ℝ | p.1 = 0} ∪ {p : ℝ × ℝ | p.2 = 0} :=
by
  sorry

end NUMINAMATH_GPT_describe_graph_l1389_138966


namespace NUMINAMATH_GPT_evaluate_expression_l1389_138926

theorem evaluate_expression : 2^(Real.log 5 / Real.log 2) + Real.log 25 / Real.log 5 = 7 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1389_138926


namespace NUMINAMATH_GPT_min_value_x_plus_y_l1389_138965

open Real

noncomputable def xy_plus_x_minus_y_minus_10_eq_zero (x y: ℝ) := x * y + x - y - 10 = 0

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : xy_plus_x_minus_y_minus_10_eq_zero x y) : 
  x + y ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_x_plus_y_l1389_138965


namespace NUMINAMATH_GPT_polynomial_value_at_2_l1389_138981

def f (x : ℝ) : ℝ := 2 * x^5 + 4 * x^4 - 2 * x^3 - 3 * x^2 + x

theorem polynomial_value_at_2 : f 2 = 102 := by
  sorry

end NUMINAMATH_GPT_polynomial_value_at_2_l1389_138981


namespace NUMINAMATH_GPT_find_x_l1389_138911

theorem find_x :
  ∃ x : ℝ, ((x * 0.85) / 2.5) - (8 * 2.25) = 5.5 ∧
  x = 69.11764705882353 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1389_138911


namespace NUMINAMATH_GPT_smallest_possible_value_l1389_138969

theorem smallest_possible_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^2 ≥ 12 * b)
  (h2 : 9 * b^2 ≥ 4 * a) : a + b = 7 :=
sorry

end NUMINAMATH_GPT_smallest_possible_value_l1389_138969


namespace NUMINAMATH_GPT_fraction_sum_equals_l1389_138987

theorem fraction_sum_equals : 
    (4 / 2) + (7 / 4) + (11 / 8) + (21 / 16) + (41 / 32) + (81 / 64) - 8 = 63 / 64 :=
by 
    sorry

end NUMINAMATH_GPT_fraction_sum_equals_l1389_138987


namespace NUMINAMATH_GPT_part_I_part_II_l1389_138917

noncomputable def vector_a : ℝ × ℝ := (4, 3)
noncomputable def vector_b : ℝ × ℝ := (5, -12)
noncomputable def vector_sum := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def vector_magnitude_sum := magnitude vector_sum
noncomputable def magnitude_a := magnitude vector_a
noncomputable def magnitude_b := magnitude vector_b
noncomputable def cos_theta := dot_product vector_a vector_b / (magnitude_a * magnitude_b)

-- Prove the magnitude of the sum of vectors is 9√2
theorem part_I : vector_magnitude_sum = 9 * Real.sqrt 2 :=
by
  sorry

-- Prove the cosine of the angle between the vectors is -16/65
theorem part_II : cos_theta = -16 / 65 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1389_138917


namespace NUMINAMATH_GPT_map_area_l1389_138901

def length : ℕ := 5
def width : ℕ := 2
def area_of_map (length width : ℕ) : ℕ := length * width

theorem map_area : area_of_map length width = 10 := by
  sorry

end NUMINAMATH_GPT_map_area_l1389_138901


namespace NUMINAMATH_GPT_women_in_room_l1389_138962

theorem women_in_room (M W : ℕ) 
  (h1 : 9 * M = 7 * W) 
  (h2 : M + 5 = 23) : 
  3 * (W - 4) = 57 :=
by
  sorry

end NUMINAMATH_GPT_women_in_room_l1389_138962


namespace NUMINAMATH_GPT_inequality_with_means_l1389_138963

theorem inequality_with_means (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := 
sorry

end NUMINAMATH_GPT_inequality_with_means_l1389_138963


namespace NUMINAMATH_GPT_am_gm_inequality_example_l1389_138929

theorem am_gm_inequality_example (x1 x2 x3 : ℝ)
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3)
  (h_sum1 : x1 + x2 + x3 = 1) :
  (x2^2 / x1) + (x3^2 / x2) + (x1^2 / x3) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_am_gm_inequality_example_l1389_138929


namespace NUMINAMATH_GPT_total_votes_is_correct_l1389_138944

-- Definitions and theorem statement
theorem total_votes_is_correct (T : ℝ) 
  (votes_for_A : ℝ) 
  (candidate_A_share : ℝ) 
  (valid_vote_fraction : ℝ) 
  (invalid_vote_fraction : ℝ) 
  (votes_for_A_equals: votes_for_A = 380800) 
  (candidate_A_share_equals: candidate_A_share = 0.80) 
  (valid_vote_fraction_equals: valid_vote_fraction = 0.85) 
  (invalid_vote_fraction_equals: invalid_vote_fraction = 0.15) 
  (valid_vote_computed: votes_for_A = candidate_A_share * valid_vote_fraction * T): 
  T = 560000 := 
by 
  sorry

end NUMINAMATH_GPT_total_votes_is_correct_l1389_138944


namespace NUMINAMATH_GPT_overall_average_score_l1389_138904

theorem overall_average_score (students_total : ℕ) (scores_day1 : ℕ) (avg1 : ℝ)
  (scores_day2 : ℕ) (avg2 : ℝ) (scores_day3 : ℕ) (avg3 : ℝ)
  (h1 : students_total = 45)
  (h2 : scores_day1 = 35)
  (h3 : avg1 = 0.65)
  (h4 : scores_day2 = 8)
  (h5 : avg2 = 0.75)
  (h6 : scores_day3 = 2)
  (h7 : avg3 = 0.85) :
  (scores_day1 * avg1 + scores_day2 * avg2 + scores_day3 * avg3) / students_total = 0.68 :=
by
  -- Lean proof goes here
  sorry

end NUMINAMATH_GPT_overall_average_score_l1389_138904


namespace NUMINAMATH_GPT_table_chair_price_l1389_138945

theorem table_chair_price
  (C T : ℝ)
  (h1 : 2 * C + T = 0.6 * (C + 2 * T))
  (h2 : T = 84) : T + C = 96 :=
sorry

end NUMINAMATH_GPT_table_chair_price_l1389_138945


namespace NUMINAMATH_GPT_malia_berries_second_bush_l1389_138919

theorem malia_berries_second_bush :
  ∀ (b2 : ℕ), ∃ (d1 d2 d3 d4 : ℕ),
  d1 = 3 → d2 = 7 → d3 = 12 → d4 = 19 →
  d2 - d1 = (d3 - d2) - 2 →
  d3 - d2 = (d4 - d3) - 2 →
  b2 = d1 + (d2 - d1 - 2) →
  b2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_malia_berries_second_bush_l1389_138919


namespace NUMINAMATH_GPT_g_f_eval_l1389_138990

def f (x : ℤ) := x^3 - 2
def g (x : ℤ) := 3 * x^2 + x + 2

theorem g_f_eval : g (f 3) = 1902 := by
  sorry

end NUMINAMATH_GPT_g_f_eval_l1389_138990


namespace NUMINAMATH_GPT_solve_for_x_l1389_138948

theorem solve_for_x : 
  ∀ x : ℝ, 
    (x ≠ 2) ∧ (4 * x^2 + 3 * x + 1) / (x - 2) = 4 * x + 5 → 
    x = -11 / 6 :=
by
  intro x
  intro h 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1389_138948


namespace NUMINAMATH_GPT_intersection_of_M_and_P_l1389_138975

def M : Set ℝ := { x | x^2 = x }
def P : Set ℝ := { x | |x - 1| = 1 }

theorem intersection_of_M_and_P : M ∩ P = {0} := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_P_l1389_138975


namespace NUMINAMATH_GPT_total_items_proof_l1389_138924

noncomputable def totalItemsBought (budget : ℕ) (sandwichCost : ℕ) 
  (pastryCost : ℕ) (maxSandwiches : ℕ) : ℕ :=
  let s := min (budget / sandwichCost) maxSandwiches
  let remainingMoney := budget - s * sandwichCost
  let p := remainingMoney / pastryCost
  s + p

theorem total_items_proof : totalItemsBought 50 6 2 7 = 11 := by
  sorry

end NUMINAMATH_GPT_total_items_proof_l1389_138924


namespace NUMINAMATH_GPT_shopkeeper_loss_l1389_138960

theorem shopkeeper_loss
    (total_stock : ℝ)
    (stock_sold_profit_percent : ℝ)
    (stock_profit_percent : ℝ)
    (stock_sold_loss_percent : ℝ)
    (stock_loss_percent : ℝ) :
    total_stock = 12500 →
    stock_sold_profit_percent = 0.20 →
    stock_profit_percent = 0.10 →
    stock_sold_loss_percent = 0.80 →
    stock_loss_percent = 0.05 →
    ∃ loss_amount, loss_amount = 250 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_loss_l1389_138960


namespace NUMINAMATH_GPT_min_red_hair_students_l1389_138947

variable (B N R : ℕ)
variable (total_students : ℕ := 50)

theorem min_red_hair_students :
  B + N + R = total_students →
  (∀ i, B > i → N > 0) →
  (∀ i, N > i → R > 0) →
  R ≥ 17 :=
by {
  -- The specifics of the proof are omitted as per the instruction
  sorry
}

end NUMINAMATH_GPT_min_red_hair_students_l1389_138947


namespace NUMINAMATH_GPT_expand_expression_l1389_138988

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l1389_138988


namespace NUMINAMATH_GPT_base_length_of_isosceles_triangle_l1389_138978

-- Definitions for the problem
def isosceles_triangle (A B C : Type) [Nonempty A] [Nonempty B] [Nonempty C] :=
  ∃ (AB BC : ℝ), AB = BC

-- The problem to prove
theorem base_length_of_isosceles_triangle
  {A B C : Type} [Nonempty A] [Nonempty B] [Nonempty C]
  (AB BC : ℝ) (AC x : ℝ)
  (height_base : ℝ) (height_side : ℝ) 
  (h1 : AB = BC)
  (h2 : height_base = 10)
  (h3 : height_side = 12)
  (h4 : AC = x)
  (h5 : ∀ AE BD : ℝ, AE = height_side → BD = height_base) :
  x = 15 := by sorry

end NUMINAMATH_GPT_base_length_of_isosceles_triangle_l1389_138978


namespace NUMINAMATH_GPT_remainder_of_3_pow_101_plus_4_mod_5_l1389_138915

theorem remainder_of_3_pow_101_plus_4_mod_5 :
  (3^101 + 4) % 5 = 2 :=
by
  have h1 : 3 % 5 = 3 := by sorry
  have h2 : (3^2) % 5 = 4 := by sorry
  have h3 : (3^3) % 5 = 2 := by sorry
  have h4 : (3^4) % 5 = 1 := by sorry
  -- more steps to show the pattern and use it to prove the final statement
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_101_plus_4_mod_5_l1389_138915


namespace NUMINAMATH_GPT_value_of_expression_l1389_138937

theorem value_of_expression (x : ℤ) (h : x = 3) : x^6 - 3 * x = 720 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1389_138937


namespace NUMINAMATH_GPT_find_numbers_l1389_138986

theorem find_numbers (a b : ℝ) (h1 : a * b = 5) (h2 : a + b = 8) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  (a = 4 + Real.sqrt 11 ∧ b = 4 - Real.sqrt 11) ∨ (a = 4 - Real.sqrt 11 ∧ b = 4 + Real.sqrt 11) :=
sorry

end NUMINAMATH_GPT_find_numbers_l1389_138986


namespace NUMINAMATH_GPT_part1_part2_l1389_138913

-- Let's define the arithmetic sequence and conditions
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + d * (n - 1)
def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Given conditions
variables (a1 a4 a3 a5 : ℕ)
variable (d : ℕ)

-- Additional conditions for the problem  
axiom h1 : a1 = 2
axiom h2 : a4 = 8
axiom h3 : arithmetic_seq a1 d 3 + arithmetic_seq a1 d 5 = a4 + 8

-- Define S7
def S7 : ℕ := sum_arithmetic_seq a1 d 7

-- Part I: Prove S7 = 56
theorem part1 : S7 = 56 := 
by
  sorry

-- Part II: Prove k = 2 given additional conditions
variable (k : ℕ)

-- Given that a_3, a_{k+1}, S_k are a geometric sequence
def is_geom_seq (a b s : ℕ) : Prop := b*b = a * s

axiom h4 : a3 = arithmetic_seq a1 d 3
axiom h5 : ∃ k, 0 < k ∧ is_geom_seq a3 (arithmetic_seq a1 d (k + 1)) (sum_arithmetic_seq a1 d k)

theorem part2 : ∃ k, 0 < k ∧ k = 2 := 
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1389_138913


namespace NUMINAMATH_GPT_trigonometric_expression_simplification_l1389_138923

theorem trigonometric_expression_simplification
  (α : ℝ) 
  (hα : α = 49 * Real.pi / 48) :
  4 * (Real.sin α ^ 3 * Real.cos (3 * α) + 
       Real.cos α ^ 3 * Real.sin (3 * α)) * 
  Real.cos (4 * α) = 0.75 := 
by 
  sorry

end NUMINAMATH_GPT_trigonometric_expression_simplification_l1389_138923


namespace NUMINAMATH_GPT_largest_quantity_l1389_138992

theorem largest_quantity (x y z w : ℤ) (h : x + 5 = y - 3 ∧ y - 3 = z + 2 ∧ z + 2 = w - 4) : w > y ∧ w > z ∧ w > x :=
by
  sorry

end NUMINAMATH_GPT_largest_quantity_l1389_138992


namespace NUMINAMATH_GPT_coupon_value_l1389_138950

theorem coupon_value (C : ℝ) (original_price : ℝ := 120) (final_price : ℝ := 99) 
(membership_discount : ℝ := 0.1) (reduced_price : ℝ := original_price - C) :
0.9 * reduced_price = final_price → C = 10 :=
by sorry

end NUMINAMATH_GPT_coupon_value_l1389_138950


namespace NUMINAMATH_GPT_average_age_combined_l1389_138900

theorem average_age_combined (n1 n2 : ℕ) (avg1 avg2 : ℕ) 
  (h1 : n1 = 45) (h2 : n2 = 60) (h3 : avg1 = 12) (h4 : avg2 = 40) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 28 :=
by
  sorry

end NUMINAMATH_GPT_average_age_combined_l1389_138900


namespace NUMINAMATH_GPT_correct_choice_option_D_l1389_138910

theorem correct_choice_option_D : (500 - 9 * 7 = 437) := by sorry

end NUMINAMATH_GPT_correct_choice_option_D_l1389_138910


namespace NUMINAMATH_GPT_find_f_of_neg3_l1389_138955

noncomputable def f : ℚ → ℚ := sorry 

theorem find_f_of_neg3 (h : ∀ (x : ℚ) (hx : x ≠ 0), 5 * f (x⁻¹) + 3 * (f x) * x⁻¹ = 2 * x^2) :
  f (-3) = -891 / 22 :=
sorry

end NUMINAMATH_GPT_find_f_of_neg3_l1389_138955


namespace NUMINAMATH_GPT_paper_pattern_after_unfolding_l1389_138954

-- Define the number of layers after folding the square paper four times
def folded_layers (initial_layers : ℕ) : ℕ :=
  initial_layers * 2 ^ 4

-- Define the number of quarter-circles removed based on the layers
def quarter_circles_removed (layers : ℕ) : ℕ :=
  layers

-- Define the number of complete circles from the quarter circles
def complete_circles (quarter_circles : ℕ) : ℕ :=
  quarter_circles / 4

-- The main theorem that we need to prove
theorem paper_pattern_after_unfolding :
  (complete_circles (quarter_circles_removed (folded_layers 1)) = 4) :=
by
  sorry

end NUMINAMATH_GPT_paper_pattern_after_unfolding_l1389_138954


namespace NUMINAMATH_GPT_max_edges_convex_polyhedron_l1389_138998

theorem max_edges_convex_polyhedron (n : ℕ) (c l e : ℕ) (h1 : c = n) (h2 : c + l = e + 2) (h3 : 2 * e ≥ 3 * l) : e ≤ 3 * n - 6 := 
sorry

end NUMINAMATH_GPT_max_edges_convex_polyhedron_l1389_138998


namespace NUMINAMATH_GPT_modular_inverse_l1389_138914

theorem modular_inverse :
  (24 * 22) % 53 = 1 :=
by
  have h1 : (24 * -29) % 53 = (53 * 0 - 29 * 24) % 53 := by sorry
  have h2 : (24 * -29) % 53 = (-29 * 24) % 53 := by sorry
  have h3 : (-29 * 24) % 53 = (-29 % 53 * 24 % 53 % 53) := by sorry
  have h4 : -29 % 53 = 53 - 24 := by sorry
  have h5 : (53 - 29) % 53 = (22 * 22) % 53 := by sorry
  have h6 : (22 * 22) % 53 = (24 * 22) % 53 := by sorry
  have h7 : (24 * 22) % 53 = 1 := by sorry
  exact h7

end NUMINAMATH_GPT_modular_inverse_l1389_138914


namespace NUMINAMATH_GPT_harry_drank_last_mile_l1389_138941

theorem harry_drank_last_mile :
  ∀ (T D start_water end_water leak_rate drink_rate leak_time first_miles : ℕ),
    start_water = 10 →
    end_water = 2 →
    leak_rate = 1 →
    leak_time = 2 →
    drink_rate = 1 →
    first_miles = 3 →
    T = leak_rate * leak_time →
    D = drink_rate * first_miles →
    start_water - end_water = T + D + (start_water - end_water - T - D) →
    start_water - end_water - T - D = 3 :=
by
  sorry

end NUMINAMATH_GPT_harry_drank_last_mile_l1389_138941
