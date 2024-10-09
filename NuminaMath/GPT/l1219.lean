import Mathlib

namespace time_3339_minutes_after_midnight_l1219_121913

def minutes_since_midnight (minutes : ℕ) : ℕ × ℕ :=
  let hours := minutes / 60
  let remaining_minutes := minutes % 60
  (hours, remaining_minutes)

def time_after_midnight (start_time : ℕ × ℕ) (hours : ℕ) (minutes : ℕ) : ℕ × ℕ :=
  let (start_hours, start_minutes) := start_time
  let total_minutes := start_hours * 60 + start_minutes + hours * 60 + minutes
  let end_hours := total_minutes / 60
  let end_minutes := total_minutes % 60
  (end_hours, end_minutes)

theorem time_3339_minutes_after_midnight :
  time_after_midnight (0, 0) 55 39 = (7, 39) :=
by
  sorry

end time_3339_minutes_after_midnight_l1219_121913


namespace total_blocks_to_ride_l1219_121990

-- Constants representing the problem conditions
def rotations_per_block : ℕ := 200
def initial_rotations : ℕ := 600
def additional_rotations : ℕ := 1000

-- Main statement asserting the total number of blocks Greg wants to ride
theorem total_blocks_to_ride : 
  (initial_rotations / rotations_per_block) + (additional_rotations / rotations_per_block) = 8 := 
  by 
    sorry

end total_blocks_to_ride_l1219_121990


namespace balls_into_boxes_l1219_121926

theorem balls_into_boxes :
  ∃ n : ℕ, n = 56 ∧ (∀ a b c d : ℕ, a + b + c + d = 5 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d →
    n = 4 * (b + c + d + 1)) :=
by sorry

end balls_into_boxes_l1219_121926


namespace distinct_positive_values_count_l1219_121918

theorem distinct_positive_values_count : 
  ∃ (n : ℕ), n = 33 ∧ ∀ (x : ℕ), 
    (20 ≤ x ∧ x ≤ 99 ∧ 20 ≤ 2 * x ∧ 2 * x < 200 ∧ 3 * x ≥ 200) 
    ↔ (67 ≤ x ∧ x < 100) :=
  sorry

end distinct_positive_values_count_l1219_121918


namespace s_of_1_l1219_121949

def t (x : ℚ) : ℚ := 5 * x - 10
def s (y : ℚ) : ℚ := (y^2 / (5^2)) + (5 * y / 5) + 6  -- reformulated to fit conditions

theorem s_of_1 :
  s (1 : ℚ) = 546 / 25 := by
  sorry

end s_of_1_l1219_121949


namespace tangent_from_point_to_circle_l1219_121914

theorem tangent_from_point_to_circle :
  ∀ (x y : ℝ),
  (x - 6)^2 + (y - 3)^2 = 4 →
  (x = 10 → y = 0 →
    4 * x - 3 * y = 19) :=
by
  sorry

end tangent_from_point_to_circle_l1219_121914


namespace determine_cans_l1219_121985

-- Definitions based on the conditions
def num_cans_total : ℕ := 140
def volume_large (y : ℝ) : ℝ := y + 2.5
def total_volume_large (x : ℕ) (y : ℝ) : ℝ := ↑x * volume_large y
def total_volume_small (x : ℕ) (y : ℝ) : ℝ := ↑(num_cans_total - x) * y

-- Proof statement
theorem determine_cans (x : ℕ) (y : ℝ) 
    (h1 : total_volume_large x y = 60)
    (h2 : total_volume_small x y = 60) : 
    x = 20 ∧ num_cans_total - x = 120 := 
by
  sorry

end determine_cans_l1219_121985


namespace find_alcohol_quantity_l1219_121952

theorem find_alcohol_quantity 
  (A W : ℝ) 
  (h1 : A / W = 2 / 5)
  (h2 : A / (W + 10) = 2 / 7) : 
  A = 10 :=
sorry

end find_alcohol_quantity_l1219_121952


namespace brady_passing_yards_proof_l1219_121948

def tom_brady_current_passing_yards 
  (record_yards : ℕ) (games_left : ℕ) (average_yards_needed : ℕ) 
  (total_yards_needed_to_break_record : ℕ :=
    record_yards + 1) : ℕ :=
  total_yards_needed_to_break_record - games_left * average_yards_needed

theorem brady_passing_yards_proof :
  tom_brady_current_passing_yards 5999 6 300 = 4200 :=
by 
  sorry

end brady_passing_yards_proof_l1219_121948


namespace B_needs_days_l1219_121923

theorem B_needs_days (A_rate B_rate Combined_rate : ℝ) (x : ℝ) (W : ℝ) (h1: A_rate = W / 140)
(h2: B_rate = W / (3 * x)) (h3 : Combined_rate = 60 * W) (h4 : Combined_rate = A_rate + B_rate) :
 x = 140 / 25197 :=
by
  sorry

end B_needs_days_l1219_121923


namespace locus_of_centers_l1219_121989

-- Statement of the problem
theorem locus_of_centers :
  ∀ (a b : ℝ),
    ((∃ r : ℝ, (a^2 + b^2 = (r + 2)^2) ∧ ((a - 1)^2 + b^2 = (3 - r)^2))) ↔ (4 * a^2 + 4 * b^2 - 25 = 0) := by
  sorry

end locus_of_centers_l1219_121989


namespace fill_trough_time_l1219_121944

theorem fill_trough_time 
  (old_pump_rate : ℝ := 1 / 600) 
  (new_pump_rate : ℝ := 1 / 200) : 
  1 / (old_pump_rate + new_pump_rate) = 150 := 
by 
  sorry

end fill_trough_time_l1219_121944


namespace skilled_new_worker_installation_avg_cost_electric_vehicle_cost_comparison_l1219_121956

-- Define the variables for the number of vehicles each type of worker can install
variables {x y : ℝ}

-- Define the conditions for system of equations
def skilled_and_new_workers_system1 (x y : ℝ) : Prop :=
  2 * x + y = 10

def skilled_and_new_workers_system2 (x y : ℝ) : Prop :=
  x + 3 * y = 10

-- Prove the number of vehicles each skilled worker and new worker can install
theorem skilled_new_worker_installation (x y : ℝ) (h1 : skilled_and_new_workers_system1 x y) (h2 : skilled_and_new_workers_system2 x y) : x = 4 ∧ y = 2 :=
by {
  -- Proof skipped
  sorry
}

-- Define the average cost equation for electric and gasoline vehicles
def avg_cost (m : ℝ) : Prop :=
  1 = 4 * (m / (m + 0.6))

-- Prove the average cost per kilometer of the electric vehicle
theorem avg_cost_electric_vehicle (m : ℝ) (h : avg_cost m) : m = 0.2 :=
by {
  -- Proof skipped
  sorry
}

-- Define annual cost equations and the comparison condition
variables {a : ℝ}
def annual_cost_electric_vehicle (a : ℝ) : ℝ :=
  0.2 * a + 6400

def annual_cost_gasoline_vehicle (a : ℝ) : ℝ :=
  0.8 * a + 4000

-- Prove that when the annual mileage is greater than 6667 kilometers, the annual cost of buying an electric vehicle is lower
theorem cost_comparison (a : ℝ) (h : a > 6667) : annual_cost_electric_vehicle a < annual_cost_gasoline_vehicle a :=
by {
  -- Proof skipped
  sorry
}

end skilled_new_worker_installation_avg_cost_electric_vehicle_cost_comparison_l1219_121956


namespace geom_seq_a5_a6_eq_180_l1219_121999

theorem geom_seq_a5_a6_eq_180 (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n+1) = a n * q)
  (cond1 : a 1 + a 2 = 20)
  (cond2 : a 3 + a 4 = 60) :
  a 5 + a 6 = 180 :=
sorry

end geom_seq_a5_a6_eq_180_l1219_121999


namespace tangent_lines_through_point_l1219_121919

theorem tangent_lines_through_point :
  ∃ k : ℚ, ((5  * k - 12 * (36 - k * 2) + 36 = 0) ∨ (2 = 0)) := sorry

end tangent_lines_through_point_l1219_121919


namespace value_of_a_l1219_121954

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem value_of_a (a : ℝ) : f' (-1) a = 4 → a = 10 / 3 := by
  sorry

end value_of_a_l1219_121954


namespace conference_end_time_correct_l1219_121930

-- Define the conference conditions
def conference_start_time : ℕ := 15 * 60 -- 3:00 p.m. in minutes
def conference_duration : ℕ := 450 -- 450 minutes duration
def daylight_saving_adjustment : ℕ := 60 -- clocks set forward by one hour

-- Define the end time computation
def end_time_without_daylight_saving : ℕ := conference_start_time + conference_duration
def end_time_with_daylight_saving : ℕ := end_time_without_daylight_saving + daylight_saving_adjustment

-- Prove that the conference ended at 11:30 p.m. (11:30 p.m. in minutes is 23 * 60 + 30)
theorem conference_end_time_correct : end_time_with_daylight_saving = 23 * 60 + 30 := by
  sorry

end conference_end_time_correct_l1219_121930


namespace time_to_cover_escalator_l1219_121908

theorem time_to_cover_escalator (escalator_speed person_speed length : ℕ) (h1 : escalator_speed = 11) (h2 : person_speed = 3) (h3 : length = 126) : 
  length / (escalator_speed + person_speed) = 9 := by
  sorry

end time_to_cover_escalator_l1219_121908


namespace koala_fiber_intake_l1219_121960

theorem koala_fiber_intake (r a : ℝ) (hr : r = 0.20) (ha : a = 8) : (a / r) = 40 :=
by
  sorry

end koala_fiber_intake_l1219_121960


namespace find_m_range_l1219_121906

variable {x y m : ℝ}

theorem find_m_range (h1 : x + 2 * y = m + 4) (h2 : 2 * x + y = 2 * m - 1)
    (h3 : x + y < 2) (h4 : x - y < 4) : m < 1 := by
  sorry

end find_m_range_l1219_121906


namespace arcade_playtime_l1219_121911

noncomputable def cost_per_six_minutes : ℝ := 0.50
noncomputable def total_spent : ℝ := 15
noncomputable def minutes_per_interval : ℝ := 6
noncomputable def minutes_per_hour : ℝ := 60

theorem arcade_playtime :
  (total_spent / cost_per_six_minutes) * minutes_per_interval / minutes_per_hour = 3 :=
by
  sorry

end arcade_playtime_l1219_121911


namespace solve_system_l1219_121921

theorem solve_system :
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ : ℚ),
  x₁ + 12 * x₂ = 15 ∧
  x₁ - 12 * x₂ + 11 * x₃ = 2 ∧
  x₁ - 11 * x₃ + 10 * x₄ = 2 ∧
  x₁ - 10 * x₄ + 9 * x₅ = 2 ∧
  x₁ - 9 * x₅ + 8 * x₆ = 2 ∧
  x₁ - 8 * x₆ + 7 * x₇ = 2 ∧
  x₁ - 7 * x₇ + 6 * x₈ = 2 ∧
  x₁ - 6 * x₈ + 5 * x₉ = 2 ∧
  x₁ - 5 * x₉ + 4 * x₁₀ = 2 ∧
  x₁ - 4 * x₁₀ + 3 * x₁₁ = 2 ∧
  x₁ - 3 * x₁₁ + 2 * x₁₂ = 2 ∧
  x₁ - 2 * x₁₂ = 2 ∧
  x₁ = 37 / 12 ∧
  x₂ = 143 / 144 ∧
  x₃ = 65 / 66 ∧
  x₄ = 39 / 40 ∧
  x₅ = 26 / 27 ∧
  x₆ = 91 / 96 ∧
  x₇ = 13 / 14 ∧
  x₈ = 65 / 72 ∧
  x₉ = 13 / 15 ∧
  x₁₀ = 13 / 16 ∧
  x₁₁ = 13 / 18 ∧
  x₁₂ = 13 / 24 :=
by
  sorry

end solve_system_l1219_121921


namespace find_a_l1219_121917

theorem find_a 
  (a : ℝ) 
  (h1 : ∀ x : ℝ, (x - 3) ^ 2 + 5 = a * x^2 + bx + c) 
  (h2 : (3, 5) = (3, a * 3 ^ 2 + b * 3 + c))
  (h3 : (-2, -20) = (-2, a * (-2)^2 + b * (-2) + c)) : a = -1 :=
by
  sorry

end find_a_l1219_121917


namespace problem_relationship_l1219_121975

theorem problem_relationship (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a :=
by {
  sorry
}

end problem_relationship_l1219_121975


namespace subtracting_seven_percent_l1219_121932

theorem subtracting_seven_percent (a : ℝ) : a - 0.07 * a = 0.93 * a :=
by 
  sorry

end subtracting_seven_percent_l1219_121932


namespace license_plates_count_l1219_121966

def numConsonantsExcludingY : Nat := 19
def numVowelsIncludingY : Nat := 6
def numConsonantsIncludingY : Nat := 21
def numEvenDigits : Nat := 5

theorem license_plates_count : 
  numConsonantsExcludingY * numVowelsIncludingY * numConsonantsIncludingY * numEvenDigits = 11970 := by
  sorry

end license_plates_count_l1219_121966


namespace car_distance_traveled_l1219_121980

theorem car_distance_traveled (d : ℝ)
  (h_avg_speed : 84.70588235294117 = 320 / ((d / 90) + (d / 80))) :
  d = 160 :=
by
  sorry

end car_distance_traveled_l1219_121980


namespace xyz_sum_56_l1219_121935

theorem xyz_sum_56 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y + z = 55) (h2 : y * z + x = 55) (h3 : z * x + y = 55)
  (even_cond : x % 2 = 0 ∨ y % 2 = 0 ∨ z % 2 = 0) :
  x + y + z = 56 :=
sorry

end xyz_sum_56_l1219_121935


namespace number_of_distinct_gardens_l1219_121993

def is_adjacent (i1 j1 i2 j2 : ℕ) : Prop :=
  (i1 = i2 ∧ (j1 = j2 + 1 ∨ j1 + 1 = j2)) ∨ 
  (j1 = j2 ∧ (i1 = i2 + 1 ∨ i1 + 1 = i2))

def is_garden (M : ℕ → ℕ → ℕ) (m n : ℕ) : Prop :=
  ∀ i j i' j', (i < m ∧ j < n ∧ i' < m ∧ j' < n ∧ is_adjacent i j i' j') → 
    ((M i j = M i' j') ∨ (M i j = M i' j' + 1) ∨ (M i j + 1 = M i' j')) ∧
  ∀ i j, (i < m ∧ j < n ∧ 
    (∀ (i' j'), is_adjacent i j i' j' → (M i j ≤ M i' j'))) → M i j = 0

theorem number_of_distinct_gardens (m n : ℕ) : 
  ∃ (count : ℕ), count = 2 ^ (m * n) - 1 :=
sorry

end number_of_distinct_gardens_l1219_121993


namespace solve_for_a_l1219_121950

variable (a b x : ℝ)

theorem solve_for_a (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) :
  a = 3 * x := sorry

end solve_for_a_l1219_121950


namespace team_B_task_alone_optimal_scheduling_l1219_121905

-- Condition definitions
def task_completed_in_18_months (A : Nat → Prop) : Prop := A 18
def work_together_complete_task_in_10_months (A B : Nat → Prop) : Prop := 
  ∃ n m : ℕ, n = 2 ∧ A n ∧ B m ∧ m = 10 ∧ ∀ x y : ℕ, (x / y = 1 / 18 + 1 / (n + 10))

-- Question 1
theorem team_B_task_alone (B : Nat → Prop) : ∃ x : ℕ, x = 27 := sorry

-- Conditions for the second theorem
def team_a_max_time (a : ℕ) : Prop := a ≤ 6
def team_b_max_time (b : ℕ) : Prop := b ≤ 24
def positive_integers (a b : ℕ) : Prop := a > 0 ∧ b > 0 
def total_work_done (a b : ℕ) : Prop := (a / 18) + (b / 27) = 1

-- Question 2
theorem optimal_scheduling (A B : Nat → Prop) : 
  ∃ a b : ℕ, team_a_max_time a ∧ team_b_max_time b ∧ positive_integers a b ∧
             (a / 18 + b / 27 = 1) → min_cost := sorry

end team_B_task_alone_optimal_scheduling_l1219_121905


namespace range_of_a_l1219_121987

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |2 * x - 1| + |x + 1| > a) ↔ a < 3 / 2 := by
  sorry

end range_of_a_l1219_121987


namespace sum_xyz_eq_neg7_l1219_121996

theorem sum_xyz_eq_neg7 (x y z : ℝ)
  (h1 : x = y + z + 2)
  (h2 : y = z + x + 1)
  (h3 : z = x + y + 4) :
  x + y + z = -7 :=
by
  sorry

end sum_xyz_eq_neg7_l1219_121996


namespace train_time_36kmph_200m_l1219_121961

/-- How many seconds will a train 200 meters long running at the rate of 36 kmph take to pass a certain telegraph post? -/
def time_to_pass_post (length_of_train : ℕ) (speed_kmph : ℕ) : ℕ :=
  length_of_train * 3600 / (speed_kmph * 1000)

theorem train_time_36kmph_200m : time_to_pass_post 200 36 = 20 := by
  sorry

end train_time_36kmph_200m_l1219_121961


namespace problem_1_problem_2_l1219_121957

-- Define f as an odd function on ℝ 
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the main property given in the problem
def property (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, (a + b) ≠ 0 → (f a + f b) / (a + b) > 0

-- Problem 1: Prove that if a > b then f(a) > f(b)
theorem problem_1 (f : ℝ → ℝ) (h_odd : odd_function f) (h_property : property f) :
  ∀ a b : ℝ, a > b → f a > f b := sorry

-- Problem 2: Prove that given f(9^x - 2 * 3^x) + f(2 * 9^x - k) > 0 for any x in [0, +∞), the range of k is k < 1
theorem problem_2 (f : ℝ → ℝ) (h_odd : odd_function f) (h_property : property f) :
  (∀ x : ℝ, 0 ≤ x → f (9 ^ x - 2 * 3 ^ x) + f (2 * 9 ^ x - k) > 0) → k < 1 := sorry

end problem_1_problem_2_l1219_121957


namespace gcd_six_digit_repeat_l1219_121939

theorem gcd_six_digit_repeat (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) : 
  ∀ m : ℕ, m = 1001 * n → (gcd m 1001 = 1001) :=
by
  sorry

end gcd_six_digit_repeat_l1219_121939


namespace tangent_line_to_curve_l1219_121997

section TangentLine

variables {x m : ℝ}

theorem tangent_line_to_curve (x0 : ℝ) :
  (∀ x : ℝ, x > 0 → y = x * Real.log x) →
  (∀ x : ℝ, y = 2 * x + m) →
  (x0 > 0) →
  (x0 * Real.log x0 = 2 * x0 + m) →
  m = -Real.exp 1 :=
by
  sorry

end TangentLine

end tangent_line_to_curve_l1219_121997


namespace new_circumference_of_circle_l1219_121984

theorem new_circumference_of_circle (w h : ℝ) (d_multiplier : ℝ) 
  (h_w : w = 7) (h_h : h = 24) (h_d_multiplier : d_multiplier = 1.5) : 
  (π * (d_multiplier * (Real.sqrt (w^2 + h^2)))) = 37.5 * π :=
by
  sorry

end new_circumference_of_circle_l1219_121984


namespace factor_expression_l1219_121927

theorem factor_expression (y z : ℝ) : 3 * y^2 - 75 * z^2 = 3 * (y + 5 * z) * (y - 5 * z) :=
by sorry

end factor_expression_l1219_121927


namespace area_of_sine_curve_l1219_121902

theorem area_of_sine_curve :
  let f := (fun x => Real.sin x)
  let a := -Real.pi
  let b := 2 * Real.pi
  (∫ x in a..b, f x) = 6 :=
by
  sorry

end area_of_sine_curve_l1219_121902


namespace farm_horses_more_than_cows_l1219_121946

variable (x : ℤ) -- number of cows initially, must be a positive integer

def initial_horses := 6 * x
def initial_cows := x
def horses_after_transaction := initial_horses - 30
def cows_after_transaction := initial_cows + 30

-- New ratio after transaction
def new_ratio := horses_after_transaction * 1 = 4 * cows_after_transaction

-- Prove that the farm owns 315 more horses than cows after transaction
theorem farm_horses_more_than_cows :
  new_ratio → horses_after_transaction - cows_after_transaction = 315 :=
by
  sorry

end farm_horses_more_than_cows_l1219_121946


namespace find_original_price_l1219_121945

-- Define the given conditions
def decreased_price : ℝ := 836
def decrease_percentage : ℝ := 0.24
def remaining_percentage : ℝ := 1 - decrease_percentage -- 76% in decimal

-- Define the original price as a variable
variable (x : ℝ)

-- State the theorem
theorem find_original_price (h : remaining_percentage * x = decreased_price) : x = 1100 :=
by
  sorry

end find_original_price_l1219_121945


namespace annual_population_increase_l1219_121974

theorem annual_population_increase (P₀ P₂ : ℝ) (r : ℝ) 
  (h0 : P₀ = 12000) 
  (h2 : P₂ = 18451.2) 
  (h_eq : P₂ = P₀ * (1 + r / 100)^2) :
  r = 24 :=
by
  sorry

end annual_population_increase_l1219_121974


namespace prove_y_minus_x_l1219_121967

theorem prove_y_minus_x (x y : ℚ) (h1 : x + y = 500) (h2 : x / y = 7 / 8) : y - x = 100 / 3 := 
by
  sorry

end prove_y_minus_x_l1219_121967


namespace relationship_abc_l1219_121962

noncomputable def a (x : ℝ) : ℝ := Real.log x
noncomputable def b (x : ℝ) : ℝ := Real.exp (Real.log x)
noncomputable def c (x : ℝ) : ℝ := Real.exp (Real.log (1 / x))

theorem relationship_abc (x : ℝ) (h : (1 / Real.exp 1) < x ∧ x < 1) : a x < b x ∧ b x < c x :=
by
  have ha : a x = Real.log x := rfl
  have hb : b x = Real.exp (Real.log x) := rfl
  have hc : c x = Real.exp (Real.log (1 / x)) := rfl
  sorry

end relationship_abc_l1219_121962


namespace bean_lands_outside_inscribed_circle_l1219_121951

theorem bean_lands_outside_inscribed_circle :
  let a := 8
  let b := 15
  let c := 17  -- hypotenuse computed as sqrt(a^2 + b^2)
  let area_triangle := (1 / 2) * a * b
  let s := (a + b + c) / 2  -- semiperimeter
  let r := area_triangle / s -- radius of the inscribed circle
  let area_incircle := π * r^2
  let probability_outside := 1 - area_incircle / area_triangle
  probability_outside = 1 - (3 * π) / 20 := 
by
  sorry

end bean_lands_outside_inscribed_circle_l1219_121951


namespace max_value_abc_l1219_121976

theorem max_value_abc (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
(h_sum : a + b + c = 3) : 
  a^2 * b^3 * c^4 ≤ 2048 / 19683 :=
sorry

end max_value_abc_l1219_121976


namespace pascal_15_5th_number_l1219_121915

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l1219_121915


namespace true_for_2_and_5_l1219_121958

theorem true_for_2_and_5 (x : ℝ) : ((x - 2) * (x - 5) = 0) ↔ (x = 2 ∨ x = 5) :=
by
  sorry

end true_for_2_and_5_l1219_121958


namespace find_c_l1219_121938

theorem find_c (a b c : ℤ) (h1 : c ≥ 0) (h2 : ¬∃ m : ℤ, 2 * a * b = m^2)
  (h3 : ∀ n : ℕ, n > 0 → (a^n + (2 : ℤ)^n) ∣ (b^n + c)) :
  c = 0 ∨ c = 1 :=
by
  sorry

end find_c_l1219_121938


namespace employed_males_percentage_l1219_121970

variables {p : ℕ} -- total population
variables {employed_p : ℕ} {employed_females_p : ℕ}

-- 60 percent of the population is employed
def employed_population (p : ℕ) : ℕ := 60 * p / 100

-- 20 percent of the employed people are females
def employed_females (employed : ℕ) : ℕ := 20 * employed / 100

-- The question we're solving:
theorem employed_males_percentage (h1 : employed_p = employed_population p)
  (h2 : employed_females_p = employed_females employed_p)
  : (employed_p - employed_females_p) * 100 / p = 48 :=
by
  sorry

end employed_males_percentage_l1219_121970


namespace lifting_ratio_after_gain_l1219_121922

def intial_lifting_total : ℕ := 2200
def initial_bodyweight : ℕ := 245
def percentage_gain_total : ℕ := 15
def weight_gain : ℕ := 8

theorem lifting_ratio_after_gain :
  (intial_lifting_total * (100 + percentage_gain_total) / 100) / (initial_bodyweight + weight_gain) = 10 := by
  sorry

end lifting_ratio_after_gain_l1219_121922


namespace danny_chemistry_marks_l1219_121965

theorem danny_chemistry_marks 
  (eng marks_physics marks_biology math : ℕ)
  (average: ℕ) 
  (total_marks: ℕ) 
  (chemistry: ℕ) 
  (h_eng : eng = 76) 
  (h_math : math = 65) 
  (h_phys : marks_physics = 82) 
  (h_bio : marks_biology = 75) 
  (h_avg : average = 73) 
  (h_total : total_marks = average * 5) : 
  chemistry = total_marks - (eng + math + marks_physics + marks_biology) :=
by
  sorry

end danny_chemistry_marks_l1219_121965


namespace total_students_end_of_year_l1219_121977

def M := 50
def E (M : ℕ) := 4 * M - 3
def H (E : ℕ) := 2 * E

def E_end (E : ℕ) := E + (E / 10)
def M_end (M : ℕ) := M - (M / 20)
def H_end (H : ℕ) := H + ((7 * H) / 100)

def total_end (E_end M_end H_end : ℕ) := E_end + M_end + H_end

theorem total_students_end_of_year : 
  total_end (E_end (E M)) (M_end M) (H_end (H (E M))) = 687 := sorry

end total_students_end_of_year_l1219_121977


namespace percent_employed_females_l1219_121916

theorem percent_employed_females (total_population employed_population employed_males : ℝ)
  (h1 : employed_population = 0.6 * total_population)
  (h2 : employed_males = 0.48 * total_population) :
  ((employed_population - employed_males) / employed_population) * 100 = 20 := 
by
  sorry

end percent_employed_females_l1219_121916


namespace units_digit_six_l1219_121934

theorem units_digit_six (n : ℕ) (h : n > 0) : (6 ^ n) % 10 = 6 :=
by sorry

example : (6 ^ 7) % 10 = 6 :=
units_digit_six 7 (by norm_num)

end units_digit_six_l1219_121934


namespace sufficient_condition_above_2c_l1219_121983

theorem sufficient_condition_above_2c (a b c : ℝ) (h1 : a > c) (h2 : b > c) : a + b > 2 * c :=
by
  sorry

end sufficient_condition_above_2c_l1219_121983


namespace quadratic_discriminant_eq_l1219_121973

theorem quadratic_discriminant_eq (a b c n : ℤ) (h_eq : a = 3) (h_b : b = -8) (h_c : c = -5)
  (h_discriminant : b^2 - 4 * a * c = n) : n = 124 := 
by
  -- proof skipped
  sorry

end quadratic_discriminant_eq_l1219_121973


namespace proportion_solution_l1219_121991

theorem proportion_solution (x : ℝ) : (x ≠ 0) → (1 / 3 = 5 / (3 * x)) → x = 5 :=
by
  intro hnx hproportion
  sorry

end proportion_solution_l1219_121991


namespace number_of_customers_left_l1219_121978

theorem number_of_customers_left (x : ℕ) (h : 14 - x + 39 = 50) : x = 3 := by
  sorry

end number_of_customers_left_l1219_121978


namespace billy_video_count_l1219_121972

theorem billy_video_count 
  (generate_suggestions : ℕ) 
  (rounds : ℕ) 
  (videos_in_total : ℕ)
  (H1 : generate_suggestions = 15)
  (H2 : rounds = 5)
  (H3 : videos_in_total = generate_suggestions * rounds + 1) : 
  videos_in_total = 76 := 
by
  sorry

end billy_video_count_l1219_121972


namespace impossible_event_l1219_121963

noncomputable def EventA := ∃ (ω : ℕ), ω = 0 ∨ ω = 1
noncomputable def EventB := ∃ (t : ℤ), t >= 0
noncomputable def Bag := {b : String // b = "White"}
noncomputable def EventC := ∀ (x : Bag), x.val ≠ "Red"
noncomputable def EventD := ∀ (a b : ℤ), (a > 0 ∧ b < 0) → a > b

theorem impossible_event:
  (EventA ∧ EventB ∧ EventD) →
  EventC :=
by
  sorry

end impossible_event_l1219_121963


namespace next_podcast_length_l1219_121969

theorem next_podcast_length 
  (drive_hours : ℕ := 6)
  (podcast1_minutes : ℕ := 45)
  (podcast2_minutes : ℕ := 90) -- Since twice the first podcast (45 * 2)
  (podcast3_minutes : ℕ := 105) -- 1 hour 45 minutes (60 + 45)
  (podcast4_minutes : ℕ := 60) -- 1 hour 
  (minutes_per_hour : ℕ := 60)
  : (drive_hours * minutes_per_hour - (podcast1_minutes + podcast2_minutes + podcast3_minutes + podcast4_minutes)) / minutes_per_hour = 1 :=
by
  sorry

end next_podcast_length_l1219_121969


namespace words_on_each_page_l1219_121912

theorem words_on_each_page (p : ℕ) (h1 : p ≤ 120) (h2 : 150 * p % 221 = 210) : p = 48 :=
sorry

end words_on_each_page_l1219_121912


namespace initial_percentage_acidic_liquid_l1219_121920

theorem initial_percentage_acidic_liquid (P : ℝ) :
  let initial_volume := 12
  let removed_volume := 4
  let final_volume := initial_volume - removed_volume
  let desired_concentration := 60
  (P/100) * initial_volume = (desired_concentration/100) * final_volume →
  P = 40 :=
by
  intros
  sorry

end initial_percentage_acidic_liquid_l1219_121920


namespace negation_prop_equiv_l1219_121931

variable (a : ℝ)

theorem negation_prop_equiv :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 - 2 * a * x - 1 < 0) ↔ (∀ x : ℝ, x > 1 → x^2 - 2 * a * x - 1 ≥ 0) :=
sorry

end negation_prop_equiv_l1219_121931


namespace problem_l1219_121981

variables {S T : ℕ → ℕ} {a b : ℕ → ℕ}

-- Conditions
-- S_n and T_n are sums of first n terms of arithmetic sequences {a_n} and {b_n}, respectively.
axiom sum_S : ∀ n, S n = n * (n + 1) / 2  -- Example: sum from 1 to n
axiom sum_T : ∀ n, T n = n * (n + 1) / 2  -- Example: sum from 1 to n

-- For any positive integer n, (S_n / T_n = (5n - 3) / (2n + 1))
axiom condition : ∀ n > 0, (S n : ℚ) / T n = (5 * n - 3 : ℚ) / (2 * n + 1)

-- Theorem to prove
theorem problem : (a 20 : ℚ) / (b 7) = 64 / 9 :=
sorry

end problem_l1219_121981


namespace pencils_per_student_l1219_121937

theorem pencils_per_student (num_students total_pencils : ℕ)
  (h1 : num_students = 4) (h2 : total_pencils = 8) : total_pencils / num_students = 2 :=
by
  -- Proof omitted
  sorry

end pencils_per_student_l1219_121937


namespace solve_for_x_l1219_121933

theorem solve_for_x (x z : ℝ) (h : z = 3 * x) :
  (4 * z^2 + z + 5 = 3 * (8 * x^2 + z + 3)) ↔ 
  (x = (1 + Real.sqrt 19) / 4 ∨ x = (1 - Real.sqrt 19) / 4) := by
  sorry

end solve_for_x_l1219_121933


namespace batsman_average_after_17th_inning_l1219_121901

-- Define the conditions and prove the required question.
theorem batsman_average_after_17th_inning (A : ℕ) (h1 : 17 * (A + 10) = 16 * A + 300) :
  (A + 10) = 140 :=
by
  sorry

end batsman_average_after_17th_inning_l1219_121901


namespace inequality_proof_l1219_121986

theorem inequality_proof (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 :=
sorry

end inequality_proof_l1219_121986


namespace elsa_data_usage_l1219_121900

theorem elsa_data_usage (D : ℝ) 
  (h_condition : D - 300 - (2/5) * (D - 300) = 120) : D = 500 := 
sorry

end elsa_data_usage_l1219_121900


namespace papaya_tree_growth_ratio_l1219_121904

theorem papaya_tree_growth_ratio :
  ∃ (a1 a2 a3 a4 a5 : ℝ),
    a1 = 2 ∧
    a2 = a1 * 1.5 ∧
    a3 = a2 * 1.5 ∧
    a4 = a3 * 2 ∧
    a1 + a2 + a3 + a4 + a5 = 23 ∧
    a5 = 4.5 ∧
    (a5 / a4) = 0.5 :=
sorry

end papaya_tree_growth_ratio_l1219_121904


namespace contrapositive_is_false_l1219_121998

-- Define the property of collinearity between two vectors
def collinear (a b : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, a = k • b

-- Define the property of vectors having the same direction
def same_direction (a b : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, k > 0 ∧ a = k • b

-- Original proposition in Lean statement
def original_proposition (a b : ℝ × ℝ) : Prop :=
  collinear a b → same_direction a b

-- Contrapositive of the original proposition
def contrapositive_proposition (a b : ℝ × ℝ) : Prop :=
  ¬ same_direction a b → ¬ collinear a b

-- The proof goal that the contrapositive is false
theorem contrapositive_is_false (a b : ℝ × ℝ) :
  (contrapositive_proposition a b) = false :=
sorry

end contrapositive_is_false_l1219_121998


namespace distance_of_intersections_l1219_121995

theorem distance_of_intersections 
  (t : ℝ)
  (x := (2 - t) * (Real.sin (Real.pi / 6)))
  (y := (-1 + t) * (Real.sin (Real.pi / 6)))
  (curve : x = y)
  (circle : x^2 + y^2 = 8) :
  ∃ (B C : ℝ × ℝ), dist B C = Real.sqrt 30 := 
by
  sorry

end distance_of_intersections_l1219_121995


namespace probability_at_least_one_die_shows_three_l1219_121979

noncomputable def probability_at_least_one_three : ℚ :=
  (15 : ℚ) / 64

theorem probability_at_least_one_die_shows_three :
  ∃ (p : ℚ), p = probability_at_least_one_three :=
by
  use (15 : ℚ) / 64
  sorry

end probability_at_least_one_die_shows_three_l1219_121979


namespace surface_area_of_equal_volume_cube_l1219_121910

def vol_rect_prism (l w h : ℝ) : ℝ := l * w * h
def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

theorem surface_area_of_equal_volume_cube :
  (vol_rect_prism 5 5 45 = surface_area_cube 10.5) :=
by
  sorry

end surface_area_of_equal_volume_cube_l1219_121910


namespace least_integer_remainder_l1219_121982

theorem least_integer_remainder (n : ℕ) 
  (h₁ : n > 1)
  (h₂ : n % 5 = 2)
  (h₃ : n % 6 = 2)
  (h₄ : n % 7 = 2)
  (h₅ : n % 8 = 2)
  (h₆ : n % 10 = 2): 
  n = 842 := 
by
  sorry

end least_integer_remainder_l1219_121982


namespace max_value_of_sinx_over_2_minus_cosx_l1219_121909

theorem max_value_of_sinx_over_2_minus_cosx (x : ℝ) : 
  ∃ y_max, y_max = (Real.sqrt 3) / 3 ∧ ∀ y, y = (Real.sin x) / (2 - Real.cos x) → y ≤ y_max :=
sorry

end max_value_of_sinx_over_2_minus_cosx_l1219_121909


namespace smallest_n_l1219_121953

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 4 * n = k^2) (h2 : ∃ l : ℕ, 5 * n = l^3) : n = 100 :=
sorry

end smallest_n_l1219_121953


namespace solve_equation_a_solve_equation_b_l1219_121936

-- Problem a
theorem solve_equation_a (a b x : ℝ) (h₀ : x ≠ a) (h₁ : x ≠ b) (h₂ : a + b ≠ 0) (h₃ : a ≠ 0) (h₄ : b ≠ 0) (h₅ : a ≠ b):
  (x + a) / (x - a) + (x + b) / (x - b) = 2 ↔ x = (2 * a * b) / (a + b) :=
by
  sorry

-- Problem b
theorem solve_equation_b (a b c d x : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : x ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) (h₅ : ab + c ≠ 0):
  c * (d / (a * b) - (a * b) / x) + d = c^2 / x ↔ x = (a * b * c) / d :=
by
  sorry

end solve_equation_a_solve_equation_b_l1219_121936


namespace sequence_general_formula_l1219_121940

noncomputable def sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2^(n-2)

theorem sequence_general_formula {a : ℕ → ℝ} {S : ℕ → ℝ} (hpos : ∀ n, a n > 0)
  (hSn : ∀ n, 2 * a n = S n + 0.5) : ∀ n, a n = sequence_formula a S n :=
by 
  sorry

end sequence_general_formula_l1219_121940


namespace negative_solutions_iff_l1219_121903

theorem negative_solutions_iff (m x y : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) :
  (x < 0 ∧ y < 0) ↔ m < -2 / 3 :=
by
  sorry

end negative_solutions_iff_l1219_121903


namespace matches_in_each_box_l1219_121994

noncomputable def matches_per_box (dozens_boxes : ℕ) (total_matches : ℕ) : ℕ :=
  total_matches / (dozens_boxes * 12)

theorem matches_in_each_box :
  matches_per_box 5 1200 = 20 :=
by
  sorry

end matches_in_each_box_l1219_121994


namespace annual_interest_rate_is_correct_l1219_121941

-- Define conditions
def principal : ℝ := 900
def finalAmount : ℝ := 992.25
def compoundingPeriods : ℕ := 2
def timeYears : ℕ := 1

-- Compound interest formula
def compound_interest (P A r : ℝ) (n t : ℕ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

-- Statement to prove
theorem annual_interest_rate_is_correct :
  ∃ r : ℝ, compound_interest principal finalAmount r compoundingPeriods timeYears ∧ r = 0.10 :=
by 
  sorry

end annual_interest_rate_is_correct_l1219_121941


namespace find_y_l1219_121968

-- Definition of the modified magic square
variable (a b c d e y : ℕ)

-- Conditions from the modified magic square problem
axiom h1 : y + 5 + c = 120 + a + c
axiom h2 : y + (y - 115) + e = 120 + b + e
axiom h3 : y + 25 + 120 = 5 + (y - 115) + (2*y - 235)

-- The statement to prove
theorem find_y : y = 245 :=
by
  sorry

end find_y_l1219_121968


namespace contrapositive_correct_l1219_121959

-- Define the main condition: If a ≥ 1/2, then ∀ x ≥ 0, f(x) ≥ 0
def main_condition (a : ℝ) (f : ℝ → ℝ) : Prop :=
  a ≥ 1/2 → ∀ x : ℝ, x ≥ 0 → f x ≥ 0

-- Define the contrapositive statement: If ∃ x ≥ 0 such that f(x) < 0, then a < 1/2
def contrapositive (a : ℝ) (f : ℝ → ℝ) : Prop :=
  (∃ x : ℝ, x ≥ 0 ∧ f x < 0) → a < 1/2

-- Theorem to prove that the contrapositive statement is correct
theorem contrapositive_correct (a : ℝ) (f : ℝ → ℝ) :
  main_condition a f ↔ contrapositive a f :=
by
  sorry

end contrapositive_correct_l1219_121959


namespace brown_gumdrops_after_replacement_l1219_121925

-- Definitions based on the given conditions.
def total_gumdrops (green_gumdrops : ℕ) : ℕ :=
  (green_gumdrops * 100) / 15

def blue_gumdrops (total_gumdrops : ℕ) : ℕ :=
  total_gumdrops * 25 / 100

def brown_gumdrops_initial (total_gumdrops : ℕ) : ℕ :=
  total_gumdrops * 15 / 100

def brown_gumdrops_final (brown_initial : ℕ) (blue_gumdrops : ℕ) : ℕ :=
  brown_initial + blue_gumdrops / 3

-- The main theorem statement based on the proof problem.
theorem brown_gumdrops_after_replacement
  (green_gumdrops : ℕ)
  (h_green : green_gumdrops = 36)
  : brown_gumdrops_final (brown_gumdrops_initial (total_gumdrops green_gumdrops)) 
                         (blue_gumdrops (total_gumdrops green_gumdrops))
    = 56 := 
  by sorry

end brown_gumdrops_after_replacement_l1219_121925


namespace prove_B_is_guilty_l1219_121943

variables (A B C : Prop)

def guilty_conditions (A B C : Prop) : Prop :=
  (A → ¬ B → C) ∧
  (C → B ∨ A) ∧
  (A → ¬ (A ∧ C)) ∧
  (A ∨ B ∨ C) ∧ 
  ¬ (¬ A ∧ ¬ B ∧ ¬ C)

theorem prove_B_is_guilty : guilty_conditions A B C → B :=
by
  intros h
  sorry

end prove_B_is_guilty_l1219_121943


namespace percent_between_20000_and_150000_l1219_121928

-- Define the percentages for each group of counties
def less_than_20000 := 30
def between_20000_and_150000 := 45
def more_than_150000 := 25

-- State the theorem using the above definitions
theorem percent_between_20000_and_150000 :
  between_20000_and_150000 = 45 :=
sorry -- Proof placeholder

end percent_between_20000_and_150000_l1219_121928


namespace cos_value_of_2alpha_plus_5pi_over_12_l1219_121988

theorem cos_value_of_2alpha_plus_5pi_over_12
  (α : ℝ) (h1 : Real.pi / 2 < α ∧ α < Real.pi)
  (h2 : Real.sin (α + Real.pi / 3) = -4 / 5) :
  Real.cos (2 * α + 5 * Real.pi / 12) = 17 * Real.sqrt 2 / 50 :=
by 
  sorry

end cos_value_of_2alpha_plus_5pi_over_12_l1219_121988


namespace initial_percentage_of_salt_l1219_121942

theorem initial_percentage_of_salt (P : ℝ) :
  (P / 100) * 80 = 8 → P = 10 :=
by
  intro h
  sorry

end initial_percentage_of_salt_l1219_121942


namespace complement_union_complement_l1219_121964

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

-- The proof problem
theorem complement_union_complement : (U \ (M ∪ N)) = {1, 6} := by
  sorry

end complement_union_complement_l1219_121964


namespace find_pairs_of_positive_integers_l1219_121992

theorem find_pairs_of_positive_integers (x y : ℕ) (h : x > 0 ∧ y > 0) (h_eq : x + y + x * y = 2006) :
  (x, y) = (2, 668) ∨ (x, y) = (668, 2) ∨ (x, y) = (8, 222) ∨ (x, y) = (222, 8) :=
sorry

end find_pairs_of_positive_integers_l1219_121992


namespace Tyler_age_l1219_121947

variable (T B S : ℕ) -- Assuming ages are non-negative integers

theorem Tyler_age (h1 : T = B - 3) (h2 : T + B + S = 25) (h3 : S = B + 2) : T = 6 := by
  sorry

end Tyler_age_l1219_121947


namespace find_k_l1219_121929

-- Define the vectors and the condition of perpendicularity
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, -1)
def c (k : ℝ) : ℝ × ℝ := (3 + k, 1 - k)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The primary statement we aim to prove
theorem find_k : ∃ k : ℝ, dot_product a (c k) = 0 ∧ k = -5 :=
by
  exists -5
  sorry

end find_k_l1219_121929


namespace fraction_irreducible_l1219_121955

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_irreducible_l1219_121955


namespace find_x_l1219_121907

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l1219_121907


namespace sqrt_domain_l1219_121971

theorem sqrt_domain (x : ℝ) : x - 5 ≥ 0 ↔ x ≥ 5 :=
by sorry

end sqrt_domain_l1219_121971


namespace sequence_general_term_l1219_121924

noncomputable def a_n (n : ℕ) : ℝ :=
  sorry

-- The main statement
theorem sequence_general_term (p q : ℝ) (hp : 0 < p) (hq : 0 < q)
  (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ m n : ℕ, |a (m + n) - a m - a n| ≤ 1 / (p * m + q * n)) :
  ∀ n : ℕ, a n = n :=
by
  sorry

end sequence_general_term_l1219_121924
