import Mathlib

namespace total_ranking_sequences_l2421_242109

-- Define teams
inductive Team
| A | B | C | D

-- Define the conditions
def qualifies (t : Team) : Prop := 
  -- Each team must win its qualifying match to participate
  true

def plays_saturday (t1 t2 t3 t4 : Team) : Prop :=
  (t1 = Team.A ∧ t2 = Team.B) ∨ (t3 = Team.C ∧ t4 = Team.D)

def plays_sunday (t1 t2 t3 t4 : Team) : Prop := 
  -- Winners of Saturday's matches play for 1st and 2nd, losers play for 3rd and 4th
  true

-- Lean statement for the proof problem
theorem total_ranking_sequences : 
  (∀ t : Team, qualifies t) → 
  (∀ t1 t2 t3 t4 : Team, plays_saturday t1 t2 t3 t4) → 
  (∀ t1 t2 t3 t4 : Team, plays_sunday t1 t2 t3 t4) → 
  ∃ n : ℕ, n = 16 :=
by 
  sorry

end total_ranking_sequences_l2421_242109


namespace percentage_return_on_investment_l2421_242128

theorem percentage_return_on_investment
  (dividend_rate : ℝ)
  (face_value : ℝ)
  (purchase_price : ℝ)
  (dividend_per_share : ℝ := (dividend_rate / 100) * face_value)
  (percentage_return : ℝ := (dividend_per_share / purchase_price) * 100)
  (h1 : dividend_rate = 15.5)
  (h2 : face_value = 50)
  (h3 : purchase_price = 31) :
  percentage_return = 25 := by
    sorry

end percentage_return_on_investment_l2421_242128


namespace intersection_A_B_l2421_242184

open Set

-- Given definitions of sets A and B
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 - 2 * x ≥ 0}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {-1, 0, 2} :=
sorry

end intersection_A_B_l2421_242184


namespace extremely_powerful_count_l2421_242129

def is_extremely_powerful (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ b % 2 = 1 ∧ a^b = n

noncomputable def count_extremely_powerful_below (m : ℕ) : ℕ :=
  Nat.card { n : ℕ | is_extremely_powerful n ∧ n < m }

theorem extremely_powerful_count : count_extremely_powerful_below 5000 = 19 :=
by
  sorry

end extremely_powerful_count_l2421_242129


namespace sum_of_angles_of_solutions_l2421_242152

theorem sum_of_angles_of_solutions : 
  ∀ (z : ℂ), z^5 = 32 * Complex.I → ∃ θs : Fin 5 → ℝ, 
  (∀ k, 0 ≤ θs k ∧ θs k < 360) ∧ (θs 0 + θs 1 + θs 2 + θs 3 + θs 4 = 810) :=
by
  sorry

end sum_of_angles_of_solutions_l2421_242152


namespace bird_weights_l2421_242143

variables (A B V G : ℕ)

theorem bird_weights : 
  A + B + V + G = 32 ∧ 
  V < G ∧ 
  V + G < B ∧ 
  A < V + B ∧ 
  G + B < A + V 
  → 
  (A = 13 ∧ V = 4 ∧ G = 5 ∧ B = 10) :=
sorry

end bird_weights_l2421_242143


namespace nine_point_circle_equation_l2421_242168

theorem nine_point_circle_equation 
  (α β γ : ℝ) 
  (x y z : ℝ) :
  (x^2 * (Real.sin α) * (Real.cos α) + y^2 * (Real.sin β) * (Real.cos β) + z^2 * (Real.sin γ) * (Real.cos γ) = 
  y * z * (Real.sin α) + x * z * (Real.sin β) + x * y * (Real.sin γ))
:= sorry

end nine_point_circle_equation_l2421_242168


namespace product_of_p_r_s_l2421_242180

-- Definition of conditions
def eq1 (p : ℕ) : Prop := 4^p + 4^3 = 320
def eq2 (r : ℕ) : Prop := 3^r + 27 = 108
def eq3 (s : ℕ) : Prop := 2^s + 7^4 = 2617

-- Main statement
theorem product_of_p_r_s (p r s : ℕ) (h1 : eq1 p) (h2 : eq2 r) (h3 : eq3 s) : p * r * s = 112 :=
by sorry

end product_of_p_r_s_l2421_242180


namespace complement_of_A_cap_B_l2421_242122

def set_A (x : ℝ) : Prop := x ≤ -4 ∨ x ≥ 2
def set_B (x : ℝ) : Prop := |x - 1| ≤ 3

def A_cap_B (x : ℝ) : Prop := set_A x ∧ set_B x

def complement_A_cap_B (x : ℝ) : Prop := ¬A_cap_B x

theorem complement_of_A_cap_B :
  {x : ℝ | complement_A_cap_B x} = {x : ℝ | x < 2 ∨ x > 4} :=
by
  sorry

end complement_of_A_cap_B_l2421_242122


namespace clea_escalator_time_l2421_242159

theorem clea_escalator_time (x y k : ℕ) (h1 : 90 * x = y) (h2 : 30 * (x + k) = y) :
  (y / k) = 45 := by
  sorry

end clea_escalator_time_l2421_242159


namespace latest_start_time_l2421_242187

-- Define the weights of the turkeys
def turkey_weights : List ℕ := [16, 18, 20, 22]

-- Define the roasting time per pound
def roasting_time_per_pound : ℕ := 15

-- Define the dinner time in 24-hour format
def dinner_time : ℕ := 18 * 60 -- 18:00 in minutes

-- Calculate the total roasting time
def total_roasting_time (weights : List ℕ) (time_per_pound : ℕ) : ℕ :=
  weights.foldr (λ weight acc => weight * time_per_pound + acc) 0

-- Calculate the latest start time
def latest_roasting_start_time (total_time : ℕ) (dinner_time : ℕ) : ℕ :=
  let start_time := dinner_time - total_time
  if start_time < 0 then start_time + 24 * 60 else start_time

-- Convert minutes to hours:minutes format
def time_in_hours_minutes (time : ℕ) : String :=
  let hours := time / 60
  let minutes := time % 60
  toString hours ++ ":" ++ toString minutes

theorem latest_start_time : 
  time_in_hours_minutes (latest_roasting_start_time (total_roasting_time turkey_weights roasting_time_per_pound) dinner_time) = "23:00" := by
  sorry

end latest_start_time_l2421_242187


namespace class_total_students_l2421_242127

-- Definitions based on the conditions
def number_students_group : ℕ := 12
def frequency_group : ℚ := 0.25

-- Statement of the problem in Lean
theorem class_total_students (n : ℕ) (h : frequency_group = number_students_group / n) : n = 48 :=
by
  sorry

end class_total_students_l2421_242127


namespace fenced_area_with_cutout_l2421_242110

theorem fenced_area_with_cutout :
  let rectangle_length : ℕ := 20
  let rectangle_width : ℕ := 16
  let cutout_length : ℕ := 4
  let cutout_width : ℕ := 4
  rectangle_length * rectangle_width - cutout_length * cutout_width = 304 := by
  sorry

end fenced_area_with_cutout_l2421_242110


namespace greatest_value_is_B_l2421_242181

def x : Int := -6

def A : Int := 2 + x
def B : Int := 2 - x
def C : Int := x - 1
def D : Int := x
def E : Int := x / 2

theorem greatest_value_is_B :
  B > A ∧ B > C ∧ B > D ∧ B > E :=
by
  sorry

end greatest_value_is_B_l2421_242181


namespace Q1_Q2_l2421_242146

noncomputable def prob_A_scores_3_out_of_4 (p_A_serves : ℚ) (p_A_scores_A_serves: ℚ) (p_A_scores_B_serves: ℚ) : ℚ :=
  by
    -- Placeholder probability function
    sorry

theorem Q1 (p_A_serves : ℚ := 2/3) (p_A_scores_A_serves: ℚ := 2/3) (p_A_scores_B_serves: ℚ := 1/2) :
  prob_A_scores_3_out_of_4 p_A_serves p_A_scores_A_serves p_A_scores_B_serves = 1/3 :=
  by
    -- Proof of the theorem
    sorry

noncomputable def prob_X_lessthan_or_equal_4 (p_A_serves: ℚ) (p_A_scores_A_serves: ℚ) (p_A_scores_B_serves: ℚ) : ℚ :=
  by
    -- Placeholder probability function
    sorry

theorem Q2 (p_A_serves: ℚ := 2/3) (p_A_scores_A_serves: ℚ := 2/3) (p_A_scores_B_serves: ℚ := 1/2) :
  prob_X_lessthan_or_equal_4 p_A_serves p_A_scores_A_serves p_A_scores_B_serves = 3/4 :=
  by
    -- Proof of the theorem
    sorry

end Q1_Q2_l2421_242146


namespace sum_a_b_l2421_242178

theorem sum_a_b (a b : ℚ) (h1 : a + 3 * b = 27) (h2 : 5 * a + 2 * b = 40) : a + b = 161 / 13 :=
  sorry

end sum_a_b_l2421_242178


namespace fraction_a_b_l2421_242113

variables {a b x y : ℝ}

theorem fraction_a_b (h1 : 4 * x - 2 * y = a) (h2 : 6 * y - 12 * x = b) (hb : b ≠ 0) :
  a / b = -1/3 := 
sorry

end fraction_a_b_l2421_242113


namespace recent_quarter_revenue_l2421_242131

theorem recent_quarter_revenue :
  let revenue_year_ago : Float := 69.0
  let percentage_decrease : Float := 30.434782608695656
  let decrease_in_revenue : Float := revenue_year_ago * (percentage_decrease / 100)
  let recent_quarter_revenue := revenue_year_ago - decrease_in_revenue
  recent_quarter_revenue = 48.0 := by
  sorry

end recent_quarter_revenue_l2421_242131


namespace mr_blue_expected_rose_petals_l2421_242126

def mr_blue_flower_bed_rose_petals (length_paces : ℕ) (width_paces : ℕ) (pace_length_ft : ℝ) (petals_per_sqft : ℝ) : ℝ :=
  let length_ft := length_paces * pace_length_ft
  let width_ft := width_paces * pace_length_ft
  let area_sqft := length_ft * width_ft
  area_sqft * petals_per_sqft

theorem mr_blue_expected_rose_petals :
  mr_blue_flower_bed_rose_petals 18 24 1.5 0.4 = 388.8 :=
by
  simp [mr_blue_flower_bed_rose_petals]
  norm_num

end mr_blue_expected_rose_petals_l2421_242126


namespace object_speed_l2421_242177

namespace problem

noncomputable def speed_in_miles_per_hour (distance_in_feet : ℕ) (time_in_seconds : ℕ) : ℝ :=
  let distance_in_miles := distance_in_feet / 5280
  let time_in_hours := time_in_seconds / 3600
  distance_in_miles / time_in_hours

theorem object_speed 
  (distance_in_feet : ℕ)
  (time_in_seconds : ℕ)
  (h : distance_in_feet = 80 ∧ time_in_seconds = 2) :
  speed_in_miles_per_hour distance_in_feet time_in_seconds = 27.27 :=
by
  sorry

end problem

end object_speed_l2421_242177


namespace trig_eq_solutions_l2421_242174

open Real

theorem trig_eq_solutions (x : ℝ) :
  2 * sin x ^ 3 + 2 * sin x ^ 2 * cos x - sin x * cos x ^ 2 - cos x ^ 3 = 0 ↔
  (∃ n : ℤ, x = -π / 4 + n * π) ∨ (∃ k : ℤ, x = arctan (sqrt 2 / 2) + k * π) ∨ (∃ m : ℤ, x = -arctan (sqrt 2 / 2) + m * π) :=
by
  sorry

end trig_eq_solutions_l2421_242174


namespace trajectory_equation_of_point_M_l2421_242102

variables {x y a b : ℝ}

theorem trajectory_equation_of_point_M :
  (a^2 + b^2 = 100) →
  (x = a / (1 + 4)) →
  (y = 4 * b / (1 + 4)) →
  16 * x^2 + y^2 = 64 :=
by
  intros h1 h2 h3
  sorry

end trajectory_equation_of_point_M_l2421_242102


namespace cost_of_song_book_l2421_242161

theorem cost_of_song_book 
  (flute_cost : ℝ) 
  (stand_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : flute_cost = 142.46) 
  (h2 : stand_cost = 8.89) 
  (h3 : total_cost = 158.35) : 
  total_cost - (flute_cost + stand_cost) = 7.00 := 
by 
  sorry

end cost_of_song_book_l2421_242161


namespace problem_1110_1111_1112_1113_l2421_242133

theorem problem_1110_1111_1112_1113 (r : ℕ) (hr : r > 5) : 
  (r^3 + r^2 + r) * (r^3 + r^2 + r + 1) * (r^3 + r^2 + r + 2) * (r^3 + r^2 + r + 3) = (r^6 + 2 * r^5 + 3 * r^4 + 5 * r^3 + 4 * r^2 + 3 * r + 1)^2 - 1 :=
by
  sorry

end problem_1110_1111_1112_1113_l2421_242133


namespace tan_alpha_is_three_halves_l2421_242191

theorem tan_alpha_is_three_halves (α : ℝ) (h : Real.tan (α - 5 * Real.pi / 4) = 1 / 5) : 
  Real.tan α = 3 / 2 :=
by
  sorry

end tan_alpha_is_three_halves_l2421_242191


namespace arithmetic_expression_value_l2421_242151

theorem arithmetic_expression_value :
  (19 + 43 / 151) * 151 = 2910 :=
by {
  sorry
}

end arithmetic_expression_value_l2421_242151


namespace trip_first_part_length_l2421_242167

theorem trip_first_part_length
  (total_distance : ℝ := 50)
  (first_speed : ℝ := 66)
  (second_speed : ℝ := 33)
  (average_speed : ℝ := 44) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ total_distance) ∧ 44 = total_distance / (x / first_speed + (total_distance - x) / second_speed) ∧ x = 25 :=
by
  sorry

end trip_first_part_length_l2421_242167


namespace necessary_but_not_sufficient_l2421_242144

variable {a b : ℝ}

theorem necessary_but_not_sufficient : (a < b + 1) ∧ ¬ (a < b + 1 → a < b) :=
by
  sorry

end necessary_but_not_sufficient_l2421_242144


namespace toys_per_week_production_l2421_242162

-- Define the necessary conditions
def days_per_week : Nat := 4
def toys_per_day : Nat := 1500

-- Define the theorem to prove the total number of toys produced per week
theorem toys_per_week_production : 
  ∀ (days_per_week toys_per_day : Nat), 
    (days_per_week = 4) →
    (toys_per_day = 1500) →
    (days_per_week * toys_per_day = 6000) := 
by
  intros
  sorry

end toys_per_week_production_l2421_242162


namespace lindas_initial_candies_l2421_242101

theorem lindas_initial_candies (candies_given : ℝ) (candies_left : ℝ) (initial_candies : ℝ) : 
  candies_given = 28 ∧ candies_left = 6 → initial_candies = candies_given + candies_left → initial_candies = 34 := 
by 
  sorry

end lindas_initial_candies_l2421_242101


namespace log_base_243_l2421_242156

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_base_243 : log_base 3 243 = 5 := by
  -- this is the statement, proof is omitted
  sorry

end log_base_243_l2421_242156


namespace dayAfter73DaysFromFridayAnd9WeeksLater_l2421_242123

-- Define the days of the week as a data type
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr

open Weekday

-- Function to calculate the day of the week after a given number of days
def addDays (start_day : Weekday) (days : ℕ) : Weekday :=
  match start_day with
  | Sunday    => match days % 7 with | 0 => Sunday    | 1 => Monday | 2 => Tuesday | 3 => Wednesday | 4 => Thursday | 5 => Friday | 6 => Saturday | _ => Sunday
  | Monday    => match days % 7 with | 0 => Monday    | 1 => Tuesday | 2 => Wednesday | 3 => Thursday | 4 => Friday | 5 => Saturday | 6 => Sunday | _ => Monday
  | Tuesday   => match days % 7 with | 0 => Tuesday   | 1 => Wednesday | 2 => Thursday | 3 => Friday | 4 => Saturday | 5 => Sunday | 6 => Monday | _ => Tuesday
  | Wednesday => match days % 7 with | 0 => Wednesday | 1 => Thursday | 2 => Friday | 3 => Saturday | 4 => Sunday | 5 => Monday | 6 => Tuesday | _ => Wednesday
  | Thursday  => match days % 7 with | 0 => Thursday  | 1 => Friday | 2 => Saturday | 3 => Sunday | 4 => Monday | 5 => Tuesday | 6 => Wednesday | _ => Thursday
  | Friday    => match days % 7 with | 0 => Friday    | 1 => Saturday | 2 => Sunday | 3 => Monday | 4 => Tuesday | 5 => Wednesday | 6 => Thursday | _ => Friday
  | Saturday  => match days % 7 with | 0 => Saturday  | 1 => Sunday | 2 => Monday | 3 => Tuesday | 4 => Wednesday | 5 => Thursday | 6 => Friday | _ => Saturday

-- Theorem that proves the required solution
theorem dayAfter73DaysFromFridayAnd9WeeksLater : addDays Friday 73 = Monday ∧ addDays Monday (9 * 7) = Monday := 
by
  -- Placeholder to acknowledge proof requirements
  sorry

end dayAfter73DaysFromFridayAnd9WeeksLater_l2421_242123


namespace inequality_proof_l2421_242172

theorem inequality_proof
  (x y : ℝ)
  (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 :=
sorry

end inequality_proof_l2421_242172


namespace value_of_a_plus_b_l2421_242116

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if 0 ≤ x then Real.sqrt x + 3 else a * x + b

theorem value_of_a_plus_b (a b : ℝ) 
  (h1 : ∀ x1 : ℝ, x1 ≠ 0 → ∃ x2 : ℝ, x1 ≠ x2 ∧ f x1 a b = f x2 a b)
  (h2 : f (2 * a) a b = f (3 * b) a b) :
  a + b = - (Real.sqrt 6) / 2 + 3 :=
by
  sorry

end value_of_a_plus_b_l2421_242116


namespace min_value_is_correct_l2421_242158

noncomputable def min_value (P : ℝ × ℝ) (A B C : ℝ × ℝ) : ℝ := 
  let PA := (A.1 - P.1, A.2 - P.2)
  let PB := (B.1 - P.1, B.2 - P.2)
  let PC := (C.1 - P.1, C.2 - P.2)
  PA.1 * PB.1 + PA.2 * PB.2 +
  PB.1 * PC.1 + PB.2 * PC.2 +
  PC.1 * PA.1 + PC.2 * PA.2

theorem min_value_is_correct :
  ∃ P : ℝ × ℝ, P = (5/3, 1/3) ∧
  min_value P (1, 4) (4, 1) (0, -4) = -62/3 :=
by
  sorry

end min_value_is_correct_l2421_242158


namespace problem1_part1_problem1_part2_l2421_242171

theorem problem1_part1 : (3 - Real.pi)^0 - 2 * Real.cos (Real.pi / 6) + abs (1 - Real.sqrt 3) + (1 / 2)⁻¹ = 2 := by
  sorry

theorem problem1_part2 {x : ℝ} : x^2 - 2 * x - 9 = 0 -> (x = 1 + Real.sqrt 10 ∨ x = 1 - Real.sqrt 10) := by
  sorry

end problem1_part1_problem1_part2_l2421_242171


namespace units_digit_of_large_power_l2421_242179

theorem units_digit_of_large_power
  (units_147_1997_pow2999: ℕ) 
  (h1 : units_147_1997_pow2999 = (147 ^ 1997) % 10)
  (h2 : ∀ k, (7 ^ (k * 4 + 1)) % 10 = 7)
  (h3 : ∀ m, (7 ^ (m * 4 + 3)) % 10 = 3)
  : units_147_1997_pow2999 % 10 = 3 :=
sorry

end units_digit_of_large_power_l2421_242179


namespace sheila_hourly_wage_l2421_242147

def sheila_works_hours : ℕ :=
  let monday_wednesday_friday := 8 * 3
  let tuesday_thursday := 6 * 2
  monday_wednesday_friday + tuesday_thursday

def sheila_weekly_earnings : ℕ := 396
def sheila_total_hours_worked := 36
def expected_hourly_earnings := sheila_weekly_earnings / sheila_total_hours_worked

theorem sheila_hourly_wage :
  sheila_works_hours = sheila_total_hours_worked ∧
  sheila_weekly_earnings / sheila_total_hours_worked = 11 :=
by
  sorry

end sheila_hourly_wage_l2421_242147


namespace squares_and_sqrt_l2421_242135

variable (a b c : ℤ)

theorem squares_and_sqrt (ha : a = 10001) (hb : b = 100010001) (hc : c = 1000200030004000300020001) :
∃ x y z : ℤ, x = a^2 ∧ y = b^2 ∧ z = Int.sqrt c ∧ x = 100020001 ∧ y = 10002000300020001 ∧ z = 1000100010001 :=
by
  use a^2, b^2, Int.sqrt c
  rw [ha, hb, hc]
  sorry

end squares_and_sqrt_l2421_242135


namespace arithmetic_problem_l2421_242115

theorem arithmetic_problem : 
  let part1 := (20 / 100) * 120
  let part2 := (25 / 100) * 250
  let part3 := (15 / 100) * 80
  let sum := part1 + part2 + part3
  let subtract := (10 / 100) * 600
  sum - subtract = 38.5 := by
  sorry

end arithmetic_problem_l2421_242115


namespace num_five_dollar_coins_l2421_242194

theorem num_five_dollar_coins (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 5 * y = 125) : y = 15 :=
by
  sorry -- Proof to be completed

end num_five_dollar_coins_l2421_242194


namespace pool_volume_l2421_242148

variable {rate1 rate2 : ℕ}
variables {hose1 hose2 hose3 hose4 : ℕ}
variables {time : ℕ}

def hose1_rate := 2
def hose2_rate := 2
def hose3_rate := 3
def hose4_rate := 3
def fill_time := 25

def total_rate := hose1_rate + hose2_rate + hose3_rate + hose4_rate

theorem pool_volume (h : hose1 = hose1_rate ∧ hose2 = hose2_rate ∧ hose3 = hose3_rate ∧ hose4 = hose4_rate ∧ time = fill_time):
  total_rate * 60 * time = 15000 := 
by 
  sorry

end pool_volume_l2421_242148


namespace pizza_store_total_sales_l2421_242185

theorem pizza_store_total_sales (pepperoni bacon cheese : ℕ) (h1 : pepperoni = 2) (h2 : bacon = 6) (h3 : cheese = 6) :
  pepperoni + bacon + cheese = 14 :=
by sorry

end pizza_store_total_sales_l2421_242185


namespace min_gx1_gx2_l2421_242169

noncomputable def f (x a : ℝ) : ℝ := x - (1 / x) - a * Real.log x
noncomputable def g (x a : ℝ) : ℝ := x - (a / 2) * Real.log x

theorem min_gx1_gx2 (x1 x2 a : ℝ) (h1 : 0 < x1 ∧ x1 < Real.exp 1) (h2 : 0 < x2) (hx1x2: x1 * x2 = 1) (ha : a > 0) :
  f x1 a = 0 ∧ f x2 a = 0 →
  g x1 a - g x2 a = -2 / Real.exp 1 :=
by sorry

end min_gx1_gx2_l2421_242169


namespace max_superior_squares_l2421_242157

theorem max_superior_squares (n : ℕ) (h : n > 2004) :
  ∃ superior_squares_count : ℕ, superior_squares_count = n * (n - 2004) := 
sorry

end max_superior_squares_l2421_242157


namespace reflect_point_l2421_242112

def point_reflect_across_line (m : ℝ) :=
  (6 - m, m + 1)

theorem reflect_point (m : ℝ) :
  point_reflect_across_line m = (6 - m, m + 1) :=
  sorry

end reflect_point_l2421_242112


namespace percentage_of_people_win_a_prize_l2421_242105

-- Define the constants used in the problem
def totalMinnows : Nat := 600
def minnowsPerPrize : Nat := 3
def totalPlayers : Nat := 800
def minnowsLeft : Nat := 240

-- Calculate the number of minnows given away as prizes
def minnowsGivenAway : Nat := totalMinnows - minnowsLeft

-- Calculate the number of prizes given away
def prizesGivenAway : Nat := minnowsGivenAway / minnowsPerPrize

-- Calculate the percentage of people winning a prize
def percentageWinners : Nat := (prizesGivenAway * 100) / totalPlayers

-- Theorem to prove the percentage of winners
theorem percentage_of_people_win_a_prize : 
    percentageWinners = 15 := 
sorry

end percentage_of_people_win_a_prize_l2421_242105


namespace total_profit_is_42000_l2421_242150

noncomputable def total_profit (I_B T_B : ℝ) :=
  let I_A := 3 * I_B
  let T_A := 2 * T_B
  let profit_B := I_B * T_B
  let profit_A := I_A * T_A
  profit_A + profit_B

theorem total_profit_is_42000
  (I_B T_B : ℝ)
  (h1 : I_A = 3 * I_B)
  (h2 : T_A = 2 * T_B)
  (h3 : I_B * T_B = 6000) :
  total_profit I_B T_B = 42000 := by
  sorry

end total_profit_is_42000_l2421_242150


namespace no_rational_solution_l2421_242130

/-- Prove that the only rational solution to the equation x^3 + 3y^3 + 9z^3 = 9xyz is x = y = z = 0. -/
theorem no_rational_solution : ∀ (x y z : ℚ), x^3 + 3 * y^3 + 9 * z^3 = 9 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro x y z h
  sorry

end no_rational_solution_l2421_242130


namespace linda_original_savings_l2421_242198

theorem linda_original_savings (S : ℝ) (h1 : 3 / 4 * S = 300 + 300) :
  S = 1200 :=
by
  sorry -- The proof is not required.

end linda_original_savings_l2421_242198


namespace max_area_rectangle_with_perimeter_40_l2421_242137

theorem max_area_rectangle_with_perimeter_40 :
  ∃ (l w : ℕ), 2 * l + 2 * w = 40 ∧ l * w = 100 :=
sorry

end max_area_rectangle_with_perimeter_40_l2421_242137


namespace least_possible_multiple_l2421_242173

theorem least_possible_multiple (x y z k : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 1 ≤ k)
  (h1 : 3 * x = k * z) (h2 : 4 * y = k * z) (h3 : x - y + z = 19) : 3 * x = 12 :=
by
  sorry

end least_possible_multiple_l2421_242173


namespace y_difference_positive_l2421_242142

theorem y_difference_positive (a c y1 y2 : ℝ) (h1 : a < 0)
  (h2 : y1 = a * 1^2 + 2 * a * 1 + c)
  (h3 : y2 = a * 2^2 + 2 * a * 2 + c) : y1 - y2 > 0 := 
sorry

end y_difference_positive_l2421_242142


namespace monotonic_intervals_max_min_values_l2421_242193

def f (x : ℝ) := x^3 - 3*x
def f_prime (x : ℝ) := 3*(x-1)*(x+1)

theorem monotonic_intervals :
  (∀ x : ℝ, x < -1 → 0 < f_prime x) ∧ (∀ x : ℝ, -1 < x ∧ x < 1 → f_prime x < 0) ∧ (∀ x : ℝ, x > 1 → 0 < f_prime x) :=
  by
  sorry

theorem max_min_values :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ 18 ∧ f x ≥ -2 ∧ 
  (f 1 = -2) ∧
  (f 3 = 18) :=
  by
  sorry

end monotonic_intervals_max_min_values_l2421_242193


namespace sequence_arithmetic_condition_l2421_242111

theorem sequence_arithmetic_condition {α β : ℝ} (hα : α ≠ 0) (hβ : β ≠ 0) (hαβ : α + β ≠ 0)
  (seq : ℕ → ℝ) (hseq : ∀ n, seq (n + 2) = (α * seq (n + 1) + β * seq n) / (α + β)) :
  ∃ α β : ℝ, (∀ a1 a2 : ℝ, α ≠ 0 ∧ β ≠ 0 ∧ α + β = 0 → seq (n + 1) - seq n = seq n - seq (n - 1)) :=
by sorry

end sequence_arithmetic_condition_l2421_242111


namespace product_of_reciprocals_l2421_242175

theorem product_of_reciprocals (x y : ℝ) (h : x + y = 6 * x * y) : (1 / x) * (1 / y) = 1 / 36 :=
by
  sorry

end product_of_reciprocals_l2421_242175


namespace least_area_of_figure_l2421_242183

theorem least_area_of_figure (c : ℝ) (hc : c > 1) : 
  ∃ A : ℝ, A = (4 / 3) * (c - 1)^(3 / 2) :=
by
  sorry

end least_area_of_figure_l2421_242183


namespace least_number_of_teams_l2421_242188

/-- A coach has 30 players in a team. If he wants to form teams of at most 7 players each for a tournament, we aim to prove that the least number of teams that he needs is 5. -/
theorem least_number_of_teams (players teams : ℕ) 
  (h_players : players = 30) 
  (h_teams : ∀ t, t ≤ 7 → t ∣ players) : teams = 5 := by
  sorry

end least_number_of_teams_l2421_242188


namespace simplify_fraction_l2421_242153

theorem simplify_fraction (x y z : ℝ) (hx : x = 5) (hz : z = 2) : (10 * x * y * z) / (15 * x^2 * z) = (2 * y) / 15 :=
by
  sorry

end simplify_fraction_l2421_242153


namespace number_of_valid_3_digit_numbers_l2421_242136

def is_even (n : ℕ) : Prop := n % 2 = 0

def valid_3_digit_numbers_count : ℕ :=
  let digits := [(4, 8), (8, 4), (6, 6)]
  digits.length * 9

theorem number_of_valid_3_digit_numbers : valid_3_digit_numbers_count = 27 :=
by
  sorry

end number_of_valid_3_digit_numbers_l2421_242136


namespace find_xyz_l2421_242149

theorem find_xyz (x y z : ℝ) :
  x - y + z = 2 ∧
  x^2 + y^2 + z^2 = 30 ∧
  x^3 - y^3 + z^3 = 116 →
  (x = -1 ∧ y = 2 ∧ z = 5) ∨
  (x = -1 ∧ y = -5 ∧ z = -2) ∨
  (x = -2 ∧ y = 1 ∧ z = 5) ∨
  (x = -2 ∧ y = -5 ∧ z = -1) ∨
  (x = 5 ∧ y = 1 ∧ z = -2) ∨
  (x = 5 ∧ y = 2 ∧ z = -1) := by
  sorry

end find_xyz_l2421_242149


namespace rabbit_probability_l2421_242189

def cube_vertices : ℕ := 8
def cube_edges : ℕ := 12
def moves : ℕ := 11
def paths_after_11_moves : ℕ := 3 ^ moves
def favorable_paths : ℕ := 24

theorem rabbit_probability :
  (favorable_paths : ℚ) / paths_after_11_moves = 24 / 177147 := by
  sorry

end rabbit_probability_l2421_242189


namespace find_y_l2421_242132

theorem find_y (y : ℕ) 
  (h : (1/8) * 2^36 = 8^y) : y = 11 :=
sorry

end find_y_l2421_242132


namespace inhabitable_land_fraction_l2421_242196

theorem inhabitable_land_fraction (total_surface not_water_covered initially_inhabitable tech_advancement_viable : ℝ)
  (h1 : not_water_covered = 1 / 3 * total_surface)
  (h2 : initially_inhabitable = 1 / 3 * not_water_covered)
  (h3 : tech_advancement_viable = 1 / 2 * (not_water_covered - initially_inhabitable)) :
  (initially_inhabitable + tech_advancement_viable) / total_surface = 2 / 9 := 
sorry

end inhabitable_land_fraction_l2421_242196


namespace width_of_lawn_is_60_l2421_242155

-- Define the problem conditions in Lean
def length_of_lawn : ℕ := 70
def road_width : ℕ := 10
def total_road_cost : ℕ := 3600
def cost_per_sq_meter : ℕ := 3

-- Define the proof problem
theorem width_of_lawn_is_60 (W : ℕ) 
  (h1 : (road_width * W) + (road_width * length_of_lawn) - (road_width * road_width) 
        = total_road_cost / cost_per_sq_meter) : 
  W = 60 := 
by 
  sorry

end width_of_lawn_is_60_l2421_242155


namespace matrix_power_problem_l2421_242166

def B : Matrix (Fin 2) (Fin 2) ℤ := 
  ![![4, 1], ![0, 2]]

theorem matrix_power_problem : B^15 - 3 * B^14 = ![![4, 3], ![0, -2]] :=
  by sorry

end matrix_power_problem_l2421_242166


namespace circle_with_all_three_colors_l2421_242106

-- Define color type using an inductive type with three colors
inductive Color
| red
| green
| blue

-- Define a function that assigns a color to each point in the plane
def color_function (point : ℝ × ℝ) : Color := sorry

-- Define the main theorem stating that for any coloring, there exists a circle that contains points of all three colors
theorem circle_with_all_three_colors (color_func : ℝ × ℝ → Color) (exists_red : ∃ p : ℝ × ℝ, color_func p = Color.red)
                                      (exists_green : ∃ p : ℝ × ℝ, color_func p = Color.green) 
                                      (exists_blue : ∃ p : ℝ × ℝ, color_func p = Color.blue) :
    ∃ (c : ℝ × ℝ) (r : ℝ), ∃ p1 p2 p3 : ℝ × ℝ, 
             color_func p1 = Color.red ∧ color_func p2 = Color.green ∧ color_func p3 = Color.blue ∧ 
             (dist p1 c = r) ∧ (dist p2 c = r) ∧ (dist p3 c = r) :=
by 
  sorry

end circle_with_all_three_colors_l2421_242106


namespace range_of_f_lt_f2_l2421_242154

-- Definitions for the given conditions
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def increasing_on (f : ℝ → ℝ) (S : Set ℝ) := ∀ ⦃a b : ℝ⦄, a ∈ S → b ∈ S → a < b → f a < f b

-- Lean 4 statement for the proof problem
theorem range_of_f_lt_f2 (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_increasing : increasing_on f {x | x ≤ 0}) : 
  ∀ x : ℝ, f x < f 2 → x > 2 ∨ x < -2 :=
by
  sorry

end range_of_f_lt_f2_l2421_242154


namespace largest_c_3_in_range_l2421_242139

theorem largest_c_3_in_range (c : ℝ) : 
  (∃ x : ℝ, x^2 - 7*x + c = 3) ↔ c ≤ 61 / 4 := 
by sorry

end largest_c_3_in_range_l2421_242139


namespace totalMoney_l2421_242125

noncomputable def joannaMoney : ℕ := 8
noncomputable def brotherMoney : ℕ := 3 * joannaMoney
noncomputable def sisterMoney : ℕ := joannaMoney / 2

theorem totalMoney : joannaMoney + brotherMoney + sisterMoney = 36 := by
  sorry

end totalMoney_l2421_242125


namespace alpha_bound_l2421_242107

theorem alpha_bound (α : ℝ) (x : ℕ → ℝ) (h_x_inc : ∀ n, x n < x (n + 1))
    (x0_one : x 0 = 1) (h_alpha : α = ∑' n, x (n + 1) / (x n)^3) :
    α ≥ 3 * Real.sqrt 3 / 2 := 
sorry

end alpha_bound_l2421_242107


namespace range_of_a_l2421_242192

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 :=
by
  sorry -- The proof is omitted as per the instructions.

end range_of_a_l2421_242192


namespace angle_Y_measure_l2421_242118

def hexagon_interior_angle_sum (n : ℕ) : ℕ :=
  180 * (n - 2)

def supplementary (α β : ℕ) : Prop :=
  α + β = 180

def equal_angles (α β γ δ : ℕ) : Prop :=
  α = β ∧ β = γ ∧ γ = δ

theorem angle_Y_measure :
  ∀ (C H E S1 S2 Y : ℕ),
    C = E ∧ E = S1 ∧ S1 = Y →
    supplementary H S2 →
    hexagon_interior_angle_sum 6 = C + H + E + S1 + S2 + Y →
    Y = 135 :=
by
  intros C H E S1 S2 Y h1 h2 h3
  sorry

end angle_Y_measure_l2421_242118


namespace number_of_feasible_networks_10_l2421_242120

-- Definitions based on conditions
def feasible_networks (n : ℕ) : ℕ :=
if n = 0 then 1 else 2 ^ (n - 1)

-- The proof problem statement
theorem number_of_feasible_networks_10 : feasible_networks 10 = 512 := by
  -- proof goes here
  sorry

end number_of_feasible_networks_10_l2421_242120


namespace min_max_values_l2421_242117

noncomputable def f (x : ℝ) : ℝ := 1 + 3 * x - x^3

theorem min_max_values : 
  (∃ x : ℝ, f x = -1) ∧ (∃ x : ℝ, f x = 3) :=
by
  sorry

end min_max_values_l2421_242117


namespace tan_15_pi_over_4_l2421_242190

theorem tan_15_pi_over_4 : Real.tan (15 * Real.pi / 4) = -1 :=
by
-- The proof is omitted.
sorry

end tan_15_pi_over_4_l2421_242190


namespace sin_cos_sum_eq_l2421_242114

theorem sin_cos_sum_eq (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.tan (θ + π / 4) = 1 / 2): 
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 := 
  sorry

end sin_cos_sum_eq_l2421_242114


namespace sum_of_solutions_l2421_242186

  theorem sum_of_solutions :
    (∃ x : ℝ, x = abs (2 * x - abs (50 - 2 * x)) ∧ ∃ y : ℝ, y = abs (2 * y - abs (50 - 2 * y)) ∧ ∃ z : ℝ, z = abs (2 * z - abs (50 - 2 * z)) ∧ (x + y + z = 170 / 3)) :=
  sorry
  
end sum_of_solutions_l2421_242186


namespace lana_spent_l2421_242134

def ticket_cost : ℕ := 6
def tickets_for_friends : ℕ := 8
def extra_tickets : ℕ := 2

theorem lana_spent :
  ticket_cost * (tickets_for_friends + extra_tickets) = 60 := 
by
  sorry

end lana_spent_l2421_242134


namespace select_people_english_japanese_l2421_242140

-- Definitions based on conditions
def total_people : ℕ := 9
def english_speakers : ℕ := 7
def japanese_speakers : ℕ := 3

-- Theorem statement
theorem select_people_english_japanese (h1 : total_people = 9) 
                                      (h2 : english_speakers = 7) 
                                      (h3 : japanese_speakers = 3) :
  ∃ n, n = 20 :=
by {
  sorry
}

end select_people_english_japanese_l2421_242140


namespace avg_of_consecutive_starting_with_b_l2421_242138

variable {a : ℕ} (h : b = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7)

theorem avg_of_consecutive_starting_with_b (h : b = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7) :
  (a + 4 + (a + 4 + 1) + (a + 4 + 2) + (a + 4 + 3) + (a + 4 + 4) + (a + 4 + 5) + (a + 4 + 6)) / 7 = a + 7 :=
  sorry

end avg_of_consecutive_starting_with_b_l2421_242138


namespace problem_solution_l2421_242119

theorem problem_solution (x : ℝ) (h : x + 1 / x = 8) : x^2 + 1 / x^2 = 62 := 
by
  sorry

end problem_solution_l2421_242119


namespace probability_of_region_D_l2421_242108

theorem probability_of_region_D
    (P_A : ℚ) (P_B : ℚ) (P_C : ℚ) (P_D : ℚ)
    (h1 : P_A = 1/4) 
    (h2 : P_B = 1/3) 
    (h3 : P_C = 1/6) 
    (h4 : P_A + P_B + P_C + P_D = 1) : 
    P_D = 1/4 := by
    sorry

end probability_of_region_D_l2421_242108


namespace math_problem_l2421_242197

variable (x y : ℝ)

theorem math_problem (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by sorry

end math_problem_l2421_242197


namespace largest_class_students_l2421_242195

theorem largest_class_students (x : ℕ)
  (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 115) : x = 27 := 
by 
  sorry

end largest_class_students_l2421_242195


namespace john_payment_l2421_242164

def total_cost (cakes : ℕ) (cost_per_cake : ℕ) : ℕ :=
  cakes * cost_per_cake

def split_cost (total : ℕ) (people : ℕ) : ℕ :=
  total / people

theorem john_payment (cakes : ℕ) (cost_per_cake : ℕ) (people : ℕ) : 
  cakes = 3 → cost_per_cake = 12 → people = 2 → 
  split_cost (total_cost cakes cost_per_cake) people = 18 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end john_payment_l2421_242164


namespace lizette_stamps_count_l2421_242100

-- Conditions
def lizette_more : ℕ := 125
def minerva_stamps : ℕ := 688

-- Proof of Lizette's stamps count
theorem lizette_stamps_count : (minerva_stamps + lizette_more = 813) :=
by 
  sorry

end lizette_stamps_count_l2421_242100


namespace tangent_line_of_circle_l2421_242176

theorem tangent_line_of_circle (x y : ℝ)
    (C_def : (x - 2)^2 + (y - 3)^2 = 25)
    (P : (ℝ × ℝ)) (P_def : P = (-1, 7)) :
    (3 * x - 4 * y + 31 = 0) :=
sorry

end tangent_line_of_circle_l2421_242176


namespace complex_series_sum_eq_zero_l2421_242160

open Complex

theorem complex_series_sum_eq_zero {ω : ℂ} (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^27 + ω^36 + ω^45 + ω^54 + ω^63 + ω^72 + ω^81 + ω^90 = 0 := by
  sorry

end complex_series_sum_eq_zero_l2421_242160


namespace acute_triangle_incorrect_option_l2421_242141

theorem acute_triangle_incorrect_option (A B C : ℝ) (hA : 0 < A ∧ A < 90) (hB : 0 < B ∧ B < 90) (hC : 0 < C ∧ C < 90)
  (angle_sum : A + B + C = 180) (h_order : A > B ∧ B > C) : ¬(B + C < 90) :=
sorry

end acute_triangle_incorrect_option_l2421_242141


namespace line_circle_no_intersection_l2421_242124

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
sorry

end line_circle_no_intersection_l2421_242124


namespace functional_eq_solution_l2421_242121

noncomputable def f : ℚ → ℚ := sorry

theorem functional_eq_solution (f : ℚ → ℚ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1):
  ∀ x : ℚ, f x = x + 1 :=
sorry

end functional_eq_solution_l2421_242121


namespace walter_age_in_2005_l2421_242182

theorem walter_age_in_2005 
  (y : ℕ) (gy : ℕ)
  (h1 : gy = 3 * y)
  (h2 : (2000 - y) + (2000 - gy) = 3896) : y + 5 = 31 :=
by {
  sorry
}

end walter_age_in_2005_l2421_242182


namespace cost_of_toys_l2421_242199

theorem cost_of_toys (x y : ℝ) (h1 : x + y = 40) (h2 : 90 / x = 150 / y) :
  x = 15 ∧ y = 25 :=
sorry

end cost_of_toys_l2421_242199


namespace extreme_values_a_1_turning_point_a_8_l2421_242145

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - (a + 2) * x + a * Real.log x

def turning_point (g : ℝ → ℝ) (P : ℝ × ℝ) (h : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), x ≠ P.1 → (g x - h x) / (x - P.1) > 0

theorem extreme_values_a_1 :
  (∀ (x : ℝ), f x 1 ≤ f (1/2) 1 → f x 1 = f (1/2) 1) ∧ (∀ (x : ℝ), f x 1 ≥ f 1 1 → f x 1 = f 1 1) :=
sorry

theorem turning_point_a_8 :
  ∀ (x₀ : ℝ), x₀ = 2 → turning_point (f · 8) (x₀, f x₀ 8) (λ x => (2 * x₀ + 8 / x₀ - 10) * (x - x₀) + x₀^2 - 10 * x₀ + 8 * Real.log x₀) :=
sorry

end extreme_values_a_1_turning_point_a_8_l2421_242145


namespace inequality_solution_l2421_242103

theorem inequality_solution (x : ℝ) : 3 * x^2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := 
sorry

end inequality_solution_l2421_242103


namespace correct_train_process_l2421_242104

-- Define each step involved in the train process
inductive Step
| buy_ticket
| wait_for_train
| check_ticket
| board_train
| repair_train

open Step

-- Define each condition as a list of steps
def process_a : List Step := [buy_ticket, wait_for_train, check_ticket, board_train]
def process_b : List Step := [wait_for_train, buy_ticket, board_train, check_ticket]
def process_c : List Step := [buy_ticket, wait_for_train, board_train, check_ticket]
def process_d : List Step := [repair_train, buy_ticket, check_ticket, board_train]

-- Define the correct process
def correct_process : List Step := [buy_ticket, wait_for_train, check_ticket, board_train]

-- The theorem to prove that process A is the correct representation
theorem correct_train_process : process_a = correct_process :=
by {
  sorry
}

end correct_train_process_l2421_242104


namespace square_perimeter_l2421_242170

theorem square_perimeter (s : ℝ) (h₁ : s^2 = 625) : 4 * s = 100 := 
sorry

end square_perimeter_l2421_242170


namespace snake_body_length_l2421_242163

theorem snake_body_length (l h : ℝ) (h_head: h = l / 10) (h_length: l = 10) : l - h = 9 := 
by 
  rw [h_length, h_head] 
  norm_num
  sorry

end snake_body_length_l2421_242163


namespace equation_is_linear_l2421_242165

-- Define the conditions and the proof statement
theorem equation_is_linear (m n : ℕ) : 3 * x ^ (2 * m + 1) - 2 * y ^ (n - 1) = 7 → (2 * m + 1 = 1) ∧ (n - 1 = 1) → m = 0 ∧ n = 2 :=
by
  sorry

end equation_is_linear_l2421_242165
