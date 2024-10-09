import Mathlib

namespace tammy_speed_on_second_day_l69_6932

variable (v₁ t₁ v₂ t₂ d₁ d₂ : ℝ)

theorem tammy_speed_on_second_day
  (h1 : t₁ + t₂ = 14)
  (h2 : t₂ = t₁ - 2)
  (h3 : d₁ + d₂ = 52)
  (h4 : v₂ = v₁ + 0.5)
  (h5 : d₁ = v₁ * t₁)
  (h6 : d₂ = v₂ * t₂)
  (h_eq : v₁ * t₁ + (v₁ + 0.5) * (t₁ - 2) = 52)
  : v₂ = 4 := 
sorry

end tammy_speed_on_second_day_l69_6932


namespace probability_rain_once_l69_6943

theorem probability_rain_once (p : ℚ) 
  (h₁ : p = 1 / 2) 
  (h₂ : 1 - p = 1 / 2) 
  (h₃ : (1 - p) ^ 4 = 1 / 16) 
  : 1 - (1 - p) ^ 4 = 15 / 16 :=
by
  sorry

end probability_rain_once_l69_6943


namespace distinct_triangles_count_l69_6950

def num_combinations (n k : ℕ) : ℕ := n.choose k

def count_collinear_sets_in_grid (grid_size : ℕ) : ℕ :=
  let rows := grid_size
  let cols := grid_size
  let diagonals := 2
  rows + cols + diagonals

noncomputable def distinct_triangles_in_grid (grid_size n k : ℕ) : ℕ :=
  num_combinations n k - count_collinear_sets_in_grid grid_size

theorem distinct_triangles_count :
  distinct_triangles_in_grid 3 9 3 = 76 := 
by 
  sorry

end distinct_triangles_count_l69_6950


namespace tan_arithmetic_sequence_l69_6988

theorem tan_arithmetic_sequence {a : ℕ → ℝ}
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + n * d)
  (h_sum : a 1 + a 7 + a 13 = Real.pi) :
  Real.tan (a 2 + a 12) = - Real.sqrt 3 :=
sorry

end tan_arithmetic_sequence_l69_6988


namespace reciprocals_not_arithmetic_sequence_l69_6975

theorem reciprocals_not_arithmetic_sequence 
  (a b c : ℝ) (h : 2 * b = a + c) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_neq : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  ¬ (1 / a + 1 / c = 2 / b) :=
by
  sorry

end reciprocals_not_arithmetic_sequence_l69_6975


namespace distinct_nonzero_reals_satisfy_equation_l69_6956

open Real

theorem distinct_nonzero_reals_satisfy_equation
  (a b c : ℝ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a) (h₄ : a ≠ 0) (h₅ : b ≠ 0) (h₆ : c ≠ 0)
  (h₇ : a + 2 / b = b + 2 / c) (h₈ : b + 2 / c = c + 2 / a) :
  (a + 2 / b) ^ 2 + (b + 2 / c) ^ 2 + (c + 2 / a) ^ 2 = 6 :=
sorry

end distinct_nonzero_reals_satisfy_equation_l69_6956


namespace rectangular_coords_transformation_l69_6927

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
(ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem rectangular_coords_transformation :
  let ρ := Real.sqrt (2 ^ 2 + (-3) ^ 2 + 6 ^ 2)
  let φ := Real.arccos (6 / ρ)
  let θ := Real.arctan (-3 / 2)
  sphericalToRectangular ρ (Real.pi + θ) φ = (-2, 3, 6) :=
by
  sorry

end rectangular_coords_transformation_l69_6927


namespace rhombus_area_correct_l69_6984

def rhombus_area (d1 d2 : ℕ) : ℕ :=
  (d1 * d2) / 2

theorem rhombus_area_correct
  (d1 d2 : ℕ)
  (h1 : d1 = 70)
  (h2 : d2 = 160) :
  rhombus_area d1 d2 = 5600 := 
by
  sorry

end rhombus_area_correct_l69_6984


namespace number_of_machines_in_first_group_l69_6971

-- Define the initial conditions
def first_group_production_rate (x : ℕ) : ℚ :=
  20 / (x * 10)

def second_group_production_rate : ℚ :=
  180 / (20 * 22.5)

-- The theorem we aim to prove
theorem number_of_machines_in_first_group (x : ℕ) (h1 : first_group_production_rate x = second_group_production_rate) :
  x = 5 :=
by
  -- Placeholder for the proof steps
  sorry

end number_of_machines_in_first_group_l69_6971


namespace length_of_tracks_l69_6920

theorem length_of_tracks (x y : ℕ) 
  (h1 : 6 * (x + 2 * y) = 5000)
  (h2 : 7 * (x + y) = 5000) : x = 5 * y :=
  sorry

end length_of_tracks_l69_6920


namespace music_track_duration_l69_6969

theorem music_track_duration (minutes : ℝ) (seconds_per_minute : ℝ) (duration_in_minutes : minutes = 12.5) (seconds_per_minute_is_60 : seconds_per_minute = 60) : minutes * seconds_per_minute = 750 := by
  sorry

end music_track_duration_l69_6969


namespace problem_l69_6926

theorem problem (n : ℝ) (h : n + 1 / n = 10) : n ^ 2 + 1 / n ^ 2 + 5 = 103 :=
by sorry

end problem_l69_6926


namespace find_y_l69_6968

theorem find_y (x y z : ℤ) (h₁ : x + y + z = 355) (h₂ : x - y = 200) (h₃ : x + z = 500) : y = -145 :=
by
  sorry

end find_y_l69_6968


namespace tan_alpha_eq_2_l69_6911

theorem tan_alpha_eq_2 (α : Real) (h : Real.tan α = 2) : 
  1 / (Real.sin (2 * α) + Real.cos (α) ^ 2) = 1 := 
by 
  sorry

end tan_alpha_eq_2_l69_6911


namespace range_of_a_l69_6972

theorem range_of_a (x : ℝ) (a : ℝ) (h₀ : x ∈ Set.Icc (-2 : ℝ) 3)
(h₁ : 2 * x - x ^ 2 ≥ a) : a ≤ 1 :=
sorry

end range_of_a_l69_6972


namespace total_cost_is_eight_x_l69_6952

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end total_cost_is_eight_x_l69_6952


namespace eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five_l69_6934

theorem eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five:
  (0.85 * 40) - (4 / 5 * 25) = 14 :=
by
  sorry

end eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five_l69_6934


namespace chloe_apples_l69_6937

theorem chloe_apples :
  ∃ x : ℕ, (∃ y : ℕ, x = y + 8 ∧ y = x / 3) ∧ x = 12 := 
by
  sorry

end chloe_apples_l69_6937


namespace necessary_but_not_sufficient_condition_l69_6973

theorem necessary_but_not_sufficient_condition (a b : ℝ) : 
  (a > b → a + 1 > b) ∧ (∃ a b : ℝ, a + 1 > b ∧ ¬ a > b) :=
by 
  sorry

end necessary_but_not_sufficient_condition_l69_6973


namespace division_of_powers_l69_6904

theorem division_of_powers :
  (0.5 ^ 4) / (0.05 ^ 3) = 500 :=
by sorry

end division_of_powers_l69_6904


namespace complement_of_union_in_U_l69_6964

-- Define the universal set U
def U : Set ℕ := {x | x < 6 ∧ x > 0}

-- Define the sets A and B
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- The complement of A ∪ B in U
def complement_U_union_A_B : Set ℕ := {x | x ∈ U ∧ x ∉ (A ∪ B)}

theorem complement_of_union_in_U : complement_U_union_A_B = {2, 4} :=
by {
  -- Placeholder for the proof
  sorry
}

end complement_of_union_in_U_l69_6964


namespace max_sum_a_b_c_d_e_f_g_l69_6991

theorem max_sum_a_b_c_d_e_f_g (a b c d e f g : ℕ)
  (h1 : a + b + c = 2)
  (h2 : b + c + d = 2)
  (h3 : c + d + e = 2)
  (h4 : d + e + f = 2)
  (h5 : e + f + g = 2) :
  a + b + c + d + e + f + g ≤ 6 := 
sorry

end max_sum_a_b_c_d_e_f_g_l69_6991


namespace negation_of_proposition_l69_6929

theorem negation_of_proposition :
  (¬ (∃ x₀ : ℝ, x₀ > 2 ∧ x₀^3 - 2 * x₀^2 < 0)) ↔ (∀ x : ℝ, x > 2 → x^3 - 2 * x^2 ≥ 0) := by
  sorry

end negation_of_proposition_l69_6929


namespace proof_problem_l69_6978

def is_solution (x : ℝ) : Prop :=
  4 * Real.cos x * Real.cos (2 * x) * Real.cos (3 * x) = Real.cos (6 * x)

noncomputable def solution (l n : ℤ) : ℝ :=
  max (Real.pi / 3 * (3 * l + 1)) (Real.pi / 4 * (2 * n + 1))

theorem proof_problem (x : ℝ) (l n : ℤ) : is_solution x → x = solution l n :=
sorry

end proof_problem_l69_6978


namespace add_neg_two_l69_6935

theorem add_neg_two : 1 + (-2 : ℚ) = -1 := by
  sorry

end add_neg_two_l69_6935


namespace max_constant_inequality_l69_6961

theorem max_constant_inequality (a b c d : ℝ) 
    (ha : 0 ≤ a) (ha1 : a ≤ 1)
    (hb : 0 ≤ b) (hb1 : b ≤ 1)
    (hc : 0 ≤ c) (hc1 : c ≤ 1)
    (hd : 0 ≤ d) (hd1 : d ≤ 1) 
    : a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^3 + b^3 + c^3 + d^3) :=
sorry

end max_constant_inequality_l69_6961


namespace choose_two_items_proof_l69_6996

   def number_of_ways_to_choose_two_items (n : ℕ) : ℕ :=
     n * (n - 1) / 2

   theorem choose_two_items_proof (n : ℕ) : number_of_ways_to_choose_two_items n = (n * (n - 1)) / 2 :=
   by
     sorry
   
end choose_two_items_proof_l69_6996


namespace find_B_values_l69_6941

theorem find_B_values (A B : ℤ) (h1 : 800 < A) (h2 : A < 1300) (h3 : B > 1) (h4 : A = B ^ 4) : B = 5 ∨ B = 6 := 
sorry

end find_B_values_l69_6941


namespace total_amount_is_70000_l69_6924

-- Definitions based on the given conditions
def total_amount_divided (amount_10: ℕ) (amount_20: ℕ) : ℕ :=
  amount_10 + amount_20

def interest_earned (amount_10: ℕ) (amount_20: ℕ) : ℕ :=
  (amount_10 * 10 / 100) + (amount_20 * 20 / 100)

-- Statement to be proved
theorem total_amount_is_70000 (amount_10: ℕ) (amount_20: ℕ) (total_interest: ℕ) :
  amount_10 = 60000 →
  total_interest = 8000 →
  interest_earned amount_10 amount_20 = total_interest →
  total_amount_divided amount_10 amount_20 = 70000 :=
by
  intros h1 h2 h3
  sorry

end total_amount_is_70000_l69_6924


namespace intersection_M_N_l69_6907

def M : Set ℝ := {x : ℝ | |x| < 1}
def N : Set ℝ := {x : ℝ | x^2 - x < 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end intersection_M_N_l69_6907


namespace find_values_l69_6908

variable (circle triangle : ℕ)

axiom condition1 : triangle = circle + circle + circle
axiom condition2 : triangle + circle = 40

theorem find_values : circle = 10 ∧ triangle = 30 :=
by
  sorry

end find_values_l69_6908


namespace total_rainfall_in_Springdale_l69_6957

theorem total_rainfall_in_Springdale
    (rainfall_first_week rainfall_second_week : ℝ)
    (h1 : rainfall_second_week = 1.5 * rainfall_first_week)
    (h2 : rainfall_second_week = 12) :
    (rainfall_first_week + rainfall_second_week = 20) :=
by
  sorry

end total_rainfall_in_Springdale_l69_6957


namespace find_m_value_l69_6999

def magic_box (a b : ℝ) : ℝ := a^2 + 2 * b - 3

theorem find_m_value (m : ℝ) :
  magic_box m (-3 * m) = 4 ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end find_m_value_l69_6999


namespace greatest_power_of_two_l69_6998

theorem greatest_power_of_two (n : ℕ) (h1 : n = 1004) (h2 : 10^n - 4^(n / 2) = k) : ∃ m : ℕ, 2 ∣ k ∧ m = 1007 :=
by
  sorry

end greatest_power_of_two_l69_6998


namespace least_number_to_add_l69_6947

theorem least_number_to_add (x : ℕ) (h : 53 ∣ x ∧ 71 ∣ x) : 
  ∃ n : ℕ, x = 1357 + n ∧ n = 2406 :=
by sorry

end least_number_to_add_l69_6947


namespace problem_l69_6982

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem problem (A_def : A = {-1, 0, 1}) : B = {0, 1} :=
by sorry

end problem_l69_6982


namespace g_is_zero_l69_6905

noncomputable def g (x : Real) : Real := 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) - 
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2)

theorem g_is_zero : ∀ x : Real, g x = 0 := by
  sorry

end g_is_zero_l69_6905


namespace find_c_l69_6915

noncomputable def f (x a b c : ℤ) := x^3 + a * x^2 + b * x + c

theorem find_c (a b c : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h₃ : f a a b c = a^3) (h₄ : f b a b c = b^3) : c = 16 :=
sorry

end find_c_l69_6915


namespace sum_of_pairwise_relatively_prime_numbers_l69_6918

theorem sum_of_pairwise_relatively_prime_numbers (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
    (h4 : a * b * c = 302400) (h5 : Nat.gcd a b = 1) (h6 : Nat.gcd b c = 1) (h7 : Nat.gcd a c = 1) :
    a + b + c = 320 :=
sorry

end sum_of_pairwise_relatively_prime_numbers_l69_6918


namespace sum_of_cubes_eq_neg2_l69_6965

theorem sum_of_cubes_eq_neg2 (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 1) : a^3 + b^3 = -2 := 
sorry

end sum_of_cubes_eq_neg2_l69_6965


namespace train_speed_l69_6902

def train_length : ℝ := 250
def bridge_length : ℝ := 150
def time_to_cross : ℝ := 32

theorem train_speed :
  (train_length + bridge_length) / time_to_cross = 12.5 :=
by {
  sorry
}

end train_speed_l69_6902


namespace sequence_expression_l69_6963

theorem sequence_expression (s a : ℕ → ℝ) (h : ∀ n : ℕ, 1 ≤ n → s n = (3 / 2 * (a n - 1))) :
  ∀ n : ℕ, 1 ≤ n → a n = 3^n :=
by
  sorry

end sequence_expression_l69_6963


namespace fan_airflow_in_one_week_l69_6985

-- Define the conditions
def fan_airflow_per_second : ℕ := 10
def fan_working_minutes_per_day : ℕ := 10
def seconds_per_minute : ℕ := 60
def days_per_week : ℕ := 7

-- Define the proof problem
theorem fan_airflow_in_one_week : (fan_airflow_per_second * fan_working_minutes_per_day * seconds_per_minute * days_per_week = 42000) := 
by sorry

end fan_airflow_in_one_week_l69_6985


namespace remainder_sum_products_l69_6940

theorem remainder_sum_products (a b c d : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) 
  (hd : d % 7 = 6) : 
  ((a * b + c * d) % 7) = 1 :=
by sorry

end remainder_sum_products_l69_6940


namespace mixed_bag_cost_l69_6958

def cost_per_pound_colombian : ℝ := 5.5
def cost_per_pound_peruvian : ℝ := 4.25
def total_weight : ℝ := 40
def weight_colombian : ℝ := 28.8

noncomputable def cost_per_pound_mixed_bag : ℝ :=
  (weight_colombian * cost_per_pound_colombian + (total_weight - weight_colombian) * cost_per_pound_peruvian) / total_weight

theorem mixed_bag_cost :
  cost_per_pound_mixed_bag = 5.15 :=
  sorry

end mixed_bag_cost_l69_6958


namespace pow_mod_remainder_l69_6928

theorem pow_mod_remainder (n : ℕ) : 5 ^ 2023 % 11 = 4 :=
by sorry

end pow_mod_remainder_l69_6928


namespace part1_part2_l69_6967

noncomputable def f (m x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem part1 (t : ℝ) :
  (1 / 2 < t ∧ t < 1) →
  (∃! t : ℝ, f 1 t = 0) := sorry

theorem part2 :
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ ∀ x : ℝ, x > 0 → f m x > 0) := sorry

end part1_part2_l69_6967


namespace sharon_trip_distance_l69_6955

noncomputable def usual_speed (x : ℝ) : ℝ := x / 200

noncomputable def reduced_speed (x : ℝ) : ℝ := x / 200 - 30 / 60

theorem sharon_trip_distance (x : ℝ) (h1 : (x / 3) / usual_speed x + (2 * x / 3) / reduced_speed x = 310) : 
x = 220 :=
by
  sorry

end sharon_trip_distance_l69_6955


namespace feb_03_2013_nine_day_l69_6912

-- Definitions of the main dates involved
def dec_21_2012 : Nat := 0  -- Assuming day 0 is Dec 21, 2012
def feb_03_2013 : Nat := 45  -- 45 days after Dec 21, 2012

-- Definition to determine the Nine-day period
def nine_day_period (x : Nat) : (Nat × Nat) :=
  let q := x / 9
  let r := x % 9
  (q + 1, r + 1)

-- Theorem we want to prove
theorem feb_03_2013_nine_day : nine_day_period feb_03_2013 = (5, 9) :=
by
  sorry

end feb_03_2013_nine_day_l69_6912


namespace average_speed_correct_l69_6948

-- Define the conditions
def distance_first_hour := 90 -- in km
def distance_second_hour := 30 -- in km
def time_first_hour := 1 -- in hours
def time_second_hour := 1 -- in hours

-- Define the total distance and total time
def total_distance := distance_first_hour + distance_second_hour
def total_time := time_first_hour + time_second_hour

-- Define the average speed
def avg_speed := total_distance / total_time

-- State the theorem to prove the average speed is 60
theorem average_speed_correct :
  avg_speed = 60 := 
by 
  -- Placeholder for the actual proof
  sorry

end average_speed_correct_l69_6948


namespace unique_x_floor_eq_20_7_l69_6977

theorem unique_x_floor_eq_20_7 : ∀ x : ℝ, (⌊x⌋ + x + 1/2 = 20.7) → x = 10.2 :=
by
  sorry

end unique_x_floor_eq_20_7_l69_6977


namespace intersection_A_B_l69_6980

open Set Real -- Opens necessary namespaces for sets and real numbers

-- Definitions for the sets A and B
def A : Set ℝ := {x | 1 / x < 1}
def B : Set ℝ := {x | x > -1}

-- The proof statement for the intersection of sets A and B
theorem intersection_A_B : A ∩ B = (Ioo (-1 : ℝ) 0) ∪ (Ioi 1) :=
by
  sorry -- Proof not included

end intersection_A_B_l69_6980


namespace min_value_of_quadratic_l69_6933

theorem min_value_of_quadratic :
  ∀ x : ℝ, ∃ z : ℝ, z = 4 * x^2 + 8 * x + 16 ∧ z ≥ 12 ∧ (∀ z' : ℝ, (z' = 4 * x^2 + 8 * x + 16) → z' ≥ 12) :=
by
  sorry

end min_value_of_quadratic_l69_6933


namespace unique_root_ln_eqn_l69_6900

/-- For what values of the parameter \(a\) does the equation
   \(\ln(x - 2a) - 3(x - 2a)^2 + 2a = 0\) have a unique root? -/
theorem unique_root_ln_eqn (a : ℝ) :
  ∃! x : ℝ, (Real.log (x - 2 * a) - 3 * (x - 2 * a) ^ 2 + 2 * a = 0) ↔
  a = (1 + Real.log 6) / 4 :=
sorry

end unique_root_ln_eqn_l69_6900


namespace problem_equiv_l69_6914

-- Definitions to match the conditions
def is_monomial (v : List ℤ) : Prop :=
  ∀ i ∈ v, True  -- Simplified; typically this would involve more specific definitions

def degree (e : String) : ℕ :=
  if e = "xy" then 2 else 0

noncomputable def coefficient (v : String) : ℤ :=
  if v = "m" then 1 else 0

-- Main fact to be proven
theorem problem_equiv :
  is_monomial [-3, 1, 5] :=
sorry

end problem_equiv_l69_6914


namespace final_lights_on_l69_6953

def lights_on_by_children : ℕ :=
  let total_lights := 200
  let flips_x := total_lights / 7
  let flips_y := total_lights / 11
  let lcm_xy := 77  -- since lcm(7, 11) = 7 * 11 = 77
  let flips_both := total_lights / lcm_xy
  flips_x + flips_y - flips_both

theorem final_lights_on : lights_on_by_children = 44 :=
by
  sorry

end final_lights_on_l69_6953


namespace cannot_obtain_fraction_3_5_l69_6936

theorem cannot_obtain_fraction_3_5 (n k : ℕ) :
  ¬ ∃ (a b : ℕ), (a = 5 + k ∧ b = 8 + k ∨ (∃ m : ℕ, a = m * 5 ∧ b = m * 8)) ∧ (a = 3 ∧ b = 5) :=
by
  sorry

end cannot_obtain_fraction_3_5_l69_6936


namespace casey_saves_money_l69_6939

def first_employee_hourly_wage : ℕ := 20
def second_employee_hourly_wage : ℕ := 22
def subsidy_per_hour : ℕ := 6
def weekly_work_hours : ℕ := 40

theorem casey_saves_money :
  let first_employee_weekly_cost := first_employee_hourly_wage * weekly_work_hours
  let second_employee_effective_hourly_wage := second_employee_hourly_wage - subsidy_per_hour
  let second_employee_weekly_cost := second_employee_effective_hourly_wage * weekly_work_hours
  let savings := first_employee_weekly_cost - second_employee_weekly_cost
  savings = 160 :=
by
  sorry

end casey_saves_money_l69_6939


namespace sum_of_constants_l69_6906

theorem sum_of_constants (x a b : ℤ) (h : x^2 - 10 * x + 15 = 0) 
    (h1 : (x + a)^2 = b) : a + b = 5 := 
sorry

end sum_of_constants_l69_6906


namespace A_B_together_l69_6954

/-- This represents the problem of finding out the number of days A and B together 
can finish a piece of work given the conditions. -/
theorem A_B_together (A_rate B_rate: ℝ) (A_days B_days: ℝ) (work: ℝ) :
  A_rate = 1 / 8 →
  A_days = 4 →
  B_rate = 1 / 12 →
  B_days = 6 →
  work = 1 →
  (A_days * A_rate + B_days * B_rate = work / 2) →
  (24 / (A_rate + B_rate) = 4.8) :=
by
  intros hA_rate hA_days hB_rate hB_days hwork hwork_done
  sorry

end A_B_together_l69_6954


namespace equal_elements_l69_6986

theorem equal_elements (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ)
  (h_perm : ∃ (σ : Equiv.Perm (Fin 2011)), ∀ i, x' i = x (σ i))
  (h_eq : ∀ i : Fin 2011, x i + x ((i + 1) % 2011) = 2 * x' i) :
  ∀ i j : Fin 2011, x i = x j :=
by
  sorry

end equal_elements_l69_6986


namespace farmer_revenue_correct_l69_6966

-- Define the conditions
def average_bacon : ℕ := 20
def price_per_pound : ℕ := 6
def size_factor : ℕ := 1 / 2

-- Calculate the bacon from the runt pig
def bacon_from_runt := average_bacon * size_factor

-- Calculate the revenue from selling the bacon
def revenue := bacon_from_runt * price_per_pound

-- Lean 4 Statement to prove
theorem farmer_revenue_correct :
  revenue = 60 :=
sorry

end farmer_revenue_correct_l69_6966


namespace range_of_a_l69_6992

theorem range_of_a (a : ℝ) :
  (∀ (x y z: ℝ), x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2 * y + 2 * z) ↔ (a ≤ -2 ∨ a ≥ 4) :=
by
sorry

end range_of_a_l69_6992


namespace total_earnings_correct_l69_6925

noncomputable def total_earnings (a_days b_days c_days b_share : ℝ) : ℝ :=
  let a_work_per_day := 1 / a_days
  let b_work_per_day := 1 / b_days
  let c_work_per_day := 1 / c_days
  let combined_work_per_day := a_work_per_day + b_work_per_day + c_work_per_day
  let b_fraction_of_total_work := b_work_per_day / combined_work_per_day
  let total_earnings := b_share / b_fraction_of_total_work
  total_earnings

theorem total_earnings_correct :
  total_earnings 6 8 12 780.0000000000001 = 2340 :=
by
  sorry

end total_earnings_correct_l69_6925


namespace equation_solutions_l69_6970

theorem equation_solutions (x : ℝ) : x * (2 * x + 1) = 2 * x + 1 ↔ x = -1 / 2 ∨ x = 1 :=
by
  sorry

end equation_solutions_l69_6970


namespace members_playing_both_l69_6938

variable (N B T Neither BT : ℕ)

theorem members_playing_both (hN : N = 30) (hB : B = 17) (hT : T = 17) (hNeither : Neither = 2) 
  (hBT : BT = B + T - (N - Neither)) : BT = 6 := 
by 
  rw [hN, hB, hT, hNeither] at hBT
  exact hBT

end members_playing_both_l69_6938


namespace inequality_proof_l69_6962

open Real

theorem inequality_proof (x y : ℝ) (hx : x > 1/2) (hy : y > 1) : 
  (4 * x^2) / (y - 1) + (y^2) / (2 * x - 1) ≥ 8 := 
by
  sorry

end inequality_proof_l69_6962


namespace sqrt_50_product_consecutive_integers_l69_6979

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l69_6979


namespace second_number_less_than_first_by_16_percent_l69_6901

variable (X : ℝ)

theorem second_number_less_than_first_by_16_percent
  (h1 : X > 0)
  (first_num : ℝ := 0.75 * X)
  (second_num : ℝ := 0.63 * X) :
  (first_num - second_num) / first_num * 100 = 16 := by
  sorry

end second_number_less_than_first_by_16_percent_l69_6901


namespace locus_of_point_T_l69_6976

theorem locus_of_point_T (r : ℝ) (a b : ℝ) (x y x1 y1 x2 y2 : ℝ)
  (hM_inside : a^2 + b^2 < r^2)
  (hK_on_circle : x1^2 + y1^2 = r^2)
  (hP_on_circle : x2^2 + y2^2 = r^2)
  (h_midpoints_eq : (x + a) / 2 = (x1 + x2) / 2 ∧ (y + b) / 2 = (y1 + y2) / 2)
  (h_diagonal_eq : (x - a)^2 + (y - b)^2 = (x1 - x2)^2 + (y1 - y2)^2) :
  x^2 + y^2 = 2 * r^2 - (a^2 + b^2) :=
  sorry

end locus_of_point_T_l69_6976


namespace find_d_l69_6919

noncomputable def single_point_graph (d : ℝ) : Prop :=
  ∃ x y : ℝ, 3 * x^2 + 2 * y^2 + 9 * x - 14 * y + d = 0

theorem find_d : single_point_graph 31.25 :=
sorry

end find_d_l69_6919


namespace remainder_is_15x_minus_14_l69_6917

noncomputable def remainder_polynomial_division : Polynomial ℝ :=
  (Polynomial.X ^ 4) % (Polynomial.X ^ 2 - 3 * Polynomial.X + 2)

theorem remainder_is_15x_minus_14 :
  remainder_polynomial_division = 15 * Polynomial.X - 14 :=
by
  sorry

end remainder_is_15x_minus_14_l69_6917


namespace inequality_proof_l69_6981

theorem inequality_proof (a b : ℝ) (x y : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_x : 0 < x) (h_y : 0 < y) : 
  (a^2 / x) + (b^2 / y) ≥ ((a + b)^2 / (x + y)) :=
sorry

end inequality_proof_l69_6981


namespace solve_equation_3x6_eq_3mx_div_xm1_l69_6974

theorem solve_equation_3x6_eq_3mx_div_xm1 (x : ℝ) 
  (h1 : x ≠ 1)
  (h2 : x^2 + 5*x - 6 ≠ 0) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) ↔ (x = 3 ∨ x = -6) :=
by 
  sorry

end solve_equation_3x6_eq_3mx_div_xm1_l69_6974


namespace total_pink_crayons_l69_6931

def mara_crayons := 40
def mara_pink_percent := 10
def luna_crayons := 50
def luna_pink_percent := 20

def pink_crayons (total_crayons : ℕ) (percent_pink : ℕ) : ℕ :=
  (percent_pink * total_crayons) / 100

def mara_pink_crayons := pink_crayons mara_crayons mara_pink_percent
def luna_pink_crayons := pink_crayons luna_crayons luna_pink_percent

theorem total_pink_crayons : mara_pink_crayons + luna_pink_crayons = 14 :=
by
  -- Proof can be written here.
  sorry

end total_pink_crayons_l69_6931


namespace infinite_primes_of_form_4n_plus_3_l69_6944

theorem infinite_primes_of_form_4n_plus_3 :
  ∀ (S : Finset ℕ), (∀ p ∈ S, Prime p ∧ p % 4 = 3) →
  ∃ q, Prime q ∧ q % 4 = 3 ∧ q ∉ S :=
by 
  sorry

end infinite_primes_of_form_4n_plus_3_l69_6944


namespace solve_system_1_solve_system_2_solve_system_3_solve_system_4_l69_6910

-- System 1
theorem solve_system_1 (x y : ℝ) (h1 : x = y + 1) (h2 : 4 * x - 3 * y = 5) : x = 2 ∧ y = 1 :=
by
  sorry

-- System 2
theorem solve_system_2 (x y : ℝ) (h1 : 3 * x + y = 8) (h2 : x - y = 4) : x = 3 ∧ y = -1 :=
by
  sorry

-- System 3
theorem solve_system_3 (x y : ℝ) (h1 : 5 * x + 3 * y = 2) (h2 : 3 * x + 2 * y = 1) : x = 1 ∧ y = -1 :=
by
  sorry

-- System 4
theorem solve_system_4 (x y z : ℝ) (h1 : x + y = 3) (h2 : y + z = -2) (h3 : z + x = 9) : x = 7 ∧ y = -4 ∧ z = 2 :=
by
  sorry

end solve_system_1_solve_system_2_solve_system_3_solve_system_4_l69_6910


namespace no_lonely_points_eventually_l69_6990

structure Graph (α : Type) :=
(vertices : Finset α)
(edges : α → Finset α)

namespace Graph

def is_lonely {α : Type} (G : Graph α) (coloring : α → Bool) (v : α) : Prop :=
  let neighbors := G.edges v
  let different_color_neighbors := neighbors.filter (λ w => coloring w ≠ coloring v)
  2 * different_color_neighbors.card > neighbors.card

end Graph

theorem no_lonely_points_eventually
  {α : Type}
  (G : Graph α)
  (initial_coloring : α → Bool) :
  ∃ (steps : Nat),
  ∀ (coloring : α → Bool),
  (∃ (t : Nat), t ≤ steps ∧ 
    (∀ v, ¬ Graph.is_lonely G coloring v)) :=
sorry

end no_lonely_points_eventually_l69_6990


namespace Jack_emails_evening_l69_6942

theorem Jack_emails_evening : 
  ∀ (morning_emails evening_emails : ℕ), 
  (morning_emails = 9) ∧ 
  (evening_emails = morning_emails - 2) → 
  evening_emails = 7 := 
by
  intros morning_emails evening_emails
  sorry

end Jack_emails_evening_l69_6942


namespace point_relationship_on_parabola_neg_x_plus_1_sq_5_l69_6909

theorem point_relationship_on_parabola_neg_x_plus_1_sq_5
  (y_1 y_2 y_3 : ℝ) :
  (A : ℝ × ℝ) = (-2, y_1) →
  (B : ℝ × ℝ) = (1, y_2) →
  (C : ℝ × ℝ) = (2, y_3) →
  (A.2 = -(A.1 + 1)^2 + 5) →
  (B.2 = -(B.1 + 1)^2 + 5) →
  (C.2 = -(C.1 + 1)^2 + 5) →
  y_1 > y_2 ∧ y_2 > y_3 :=
by
  sorry

end point_relationship_on_parabola_neg_x_plus_1_sq_5_l69_6909


namespace value_of_8b_l69_6959

theorem value_of_8b (a b : ℝ) (h1 : 6 * a + 3 * b = 3) (h2 : b = 2 * a - 3) : 8 * b = -8 := by
  sorry

end value_of_8b_l69_6959


namespace smaller_rectangle_perimeter_l69_6945

def perimeter_original_rectangle (a b : ℝ) : Prop := 2 * (a + b) = 100
def number_of_cuts (vertical_cuts horizontal_cuts : ℕ) : Prop := vertical_cuts = 7 ∧ horizontal_cuts = 10
def total_length_of_cuts (a b : ℝ) : Prop := 7 * b + 10 * a = 434

theorem smaller_rectangle_perimeter (a b : ℝ) (vertical_cuts horizontal_cuts : ℕ) (m n : ℕ) :
  perimeter_original_rectangle a b →
  number_of_cuts vertical_cuts horizontal_cuts →
  total_length_of_cuts a b →
  (m = 8) →
  (n = 11) →
  (a / 8 + b / 11) * 2 = 11 :=
by
  sorry

end smaller_rectangle_perimeter_l69_6945


namespace find_functions_l69_6913

theorem find_functions (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2002 * x - f 0) = 2002 * x^2) :
  (∀ x, f x = (x^2) / 2002) ∨ (∀ x, f x = (x^2) / 2002 + 2 * x + 2002) :=
sorry

end find_functions_l69_6913


namespace base6_divisibility_13_l69_6951

theorem base6_divisibility_13 (d : ℕ) (h : 0 ≤ d ∧ d ≤ 5) : (435 + 42 * d) % 13 = 0 ↔ d = 5 :=
by sorry

end base6_divisibility_13_l69_6951


namespace train_length_l69_6993

theorem train_length (T : ℕ) (S : ℕ) (conversion_factor : ℚ) (L : ℕ) 
  (hT : T = 16)
  (hS : S = 108)
  (hconv : conversion_factor = 5 / 18)
  (hL : L = 480) :
  L = ((S * conversion_factor : ℚ) * T : ℚ) :=
sorry

end train_length_l69_6993


namespace distance_between_A_and_B_l69_6989

theorem distance_between_A_and_B (v_A v_B d d' : ℝ)
  (h1 : v_B = 50)
  (h2 : (v_A - v_B) * 30 = d')
  (h3 : (v_A + v_B) * 6 = d) :
  d = 750 :=
sorry

end distance_between_A_and_B_l69_6989


namespace julie_aaron_age_l69_6916

variables {J A m : ℕ}

theorem julie_aaron_age : (J = 4 * A) → (J + 10 = m * (A + 10)) → (m = 4) :=
by
  intros h1 h2
  sorry

end julie_aaron_age_l69_6916


namespace Mary_works_hours_on_Tuesday_and_Thursday_l69_6903

theorem Mary_works_hours_on_Tuesday_and_Thursday 
  (h_mon_wed_fri : ∀ (d : ℕ), d = 3 → 9 * d = 27)
  (weekly_earnings : ℕ)
  (hourly_rate : ℕ)
  (weekly_hours_mon_wed_fri : ℕ)
  (tue_thu_hours : ℕ) :
  weekly_earnings = 407 →
  hourly_rate = 11 →
  weekly_hours_mon_wed_fri = 9 * 3 →
  weekly_earnings - weekly_hours_mon_wed_fri * hourly_rate = tue_thu_hours * hourly_rate →
  tue_thu_hours = 10 :=
by
  intros hearnings hrate hweek hsub
  sorry

end Mary_works_hours_on_Tuesday_and_Thursday_l69_6903


namespace find_m_n_l69_6923

theorem find_m_n (m n : ℝ) : (∀ x : ℝ, -5 ≤ x ∧ x ≤ 1 → x^2 - m * x + n ≤ 0) → m = -4 ∧ n = -5 :=
by
  sorry

end find_m_n_l69_6923


namespace multiples_of_10_5_l69_6997

theorem multiples_of_10_5 (n : ℤ) (h1 : ∀ k : ℤ, k % 10 = 0 → k % 5 = 0) (h2 : n % 10 = 0) : n % 5 = 0 := 
by
  sorry

end multiples_of_10_5_l69_6997


namespace margaret_time_l69_6983

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

def total_permutations (n : Nat) : Nat :=
  factorial n

def total_time_in_minutes (total_permutations : Nat) (rate : Nat) : Nat :=
  total_permutations / rate

def time_in_hours_and_minutes (total_minutes : Nat) : Nat × Nat :=
  let hours := total_minutes / 60
  let minutes := total_minutes % 60
  (hours, minutes)

theorem margaret_time :
  let n := 8
  let r := 15
  let permutations := total_permutations n
  let total_minutes := total_time_in_minutes permutations r
  time_in_hours_and_minutes total_minutes = (44, 48) := by
  sorry

end margaret_time_l69_6983


namespace subtracted_value_l69_6922

theorem subtracted_value (s : ℕ) (h : s = 4) (x : ℕ) (h2 : (s + s^2 - x = 4)) : x = 16 :=
by
  sorry

end subtracted_value_l69_6922


namespace number_of_rows_l69_6946

theorem number_of_rows (total_chairs : ℕ) (chairs_per_row : ℕ) (r : ℕ) 
  (h1 : total_chairs = 432) (h2 : chairs_per_row = 16) (h3 : total_chairs = chairs_per_row * r) : r = 27 :=
sorry

end number_of_rows_l69_6946


namespace total_guests_l69_6960

-- Define the conditions.
def number_of_tables := 252.0
def guests_per_table := 4.0

-- Define the statement to prove.
theorem total_guests : number_of_tables * guests_per_table = 1008.0 := by
  sorry

end total_guests_l69_6960


namespace larger_factor_of_lcm_l69_6995

theorem larger_factor_of_lcm (A B : ℕ) (hcf lcm X Y : ℕ) 
  (h_hcf: hcf = 63)
  (h_A: A = 1071)
  (h_lcm: lcm = hcf * X * Y)
  (h_X: X = 11)
  (h_factors: ∃ k: ℕ, A = hcf * k ∧ lcm = A * (B / k)):
  Y = 17 := 
by sorry

end larger_factor_of_lcm_l69_6995


namespace frigate_catches_smuggler_at_five_l69_6921

noncomputable def time_to_catch : ℝ :=
  2 + (12 / 4) -- Initial leading distance / Relative speed before storm
  
theorem frigate_catches_smuggler_at_five 
  (initial_distance : ℝ)
  (frigate_speed_before_storm : ℝ)
  (smuggler_speed_before_storm : ℝ)
  (time_before_storm : ℝ)
  (frigate_speed_after_storm : ℝ)
  (smuggler_speed_after_storm : ℝ) :
  initial_distance = 12 →
  frigate_speed_before_storm = 14 →
  smuggler_speed_before_storm = 10 →
  time_before_storm = 3 →
  frigate_speed_after_storm = 12 →
  smuggler_speed_after_storm = 9 →
  time_to_catch = 5 :=
by
{
  sorry
}

end frigate_catches_smuggler_at_five_l69_6921


namespace least_z_minus_x_l69_6987

theorem least_z_minus_x (x y z : ℤ) (h1 : x < y) (h2 : y < z) (h3 : y - x > 3) (h4 : Even x) (h5 : Odd y) (h6 : Odd z) : z - x = 7 :=
sorry

end least_z_minus_x_l69_6987


namespace simplify_fraction_l69_6994

theorem simplify_fraction (a : ℝ) (h : a = 2) : (15 * a^4) / (75 * a^3) = 2 / 5 :=
by
  sorry

end simplify_fraction_l69_6994


namespace SharonOranges_l69_6949

-- Define the given conditions
def JanetOranges : Nat := 9
def TotalOranges : Nat := 16

-- Define the statement that needs to be proven
theorem SharonOranges (J : Nat) (T : Nat) (S : Nat) (hJ : J = 9) (hT : T = 16) (hS : S = T - J) : S = 7 := by
  -- (proof to be filled in later)
  sorry

end SharonOranges_l69_6949


namespace electricity_cost_one_kilometer_minimum_electricity_kilometers_l69_6930

-- Part 1: Cost of traveling one kilometer using electricity only
theorem electricity_cost_one_kilometer (x : ℝ) (fuel_cost : ℝ) (electricity_cost : ℝ) 
  (total_fuel_cost : ℝ) (total_electricity_cost : ℝ) 
  (fuel_per_km_more_than_electricity : ℝ) (distance_fuel : ℝ) (distance_electricity : ℝ)
  (h1 : total_fuel_cost = distance_fuel * fuel_cost)
  (h2 : total_electricity_cost = distance_electricity * electricity_cost)
  (h3 : fuel_per_km_more_than_electricity = 0.5)
  (h4 : fuel_cost = electricity_cost + fuel_per_km_more_than_electricity)
  (h5 : distance_fuel = 76 / (electricity_cost + 0.5))
  (h6 : distance_electricity = 26 / electricity_cost) : 
  x = 0.26 :=
sorry

-- Part 2: Minimum kilometers traveled using electricity
theorem minimum_electricity_kilometers (total_trip_cost : ℝ) (electricity_per_km : ℝ) 
  (hybrid_total_km : ℝ) (max_total_cost : ℝ) (fuel_per_km : ℝ) (y : ℝ)
  (h1 : electricity_per_km = 0.26)
  (h2 : fuel_per_km = 0.26 + 0.5)
  (h3 : hybrid_total_km = 100)
  (h4 : max_total_cost = 39)
  (h5 : total_trip_cost = electricity_per_km * y + (hybrid_total_km - y) * fuel_per_km)
  (h6 : total_trip_cost ≤ max_total_cost) :
  y ≥ 74 :=
sorry

end electricity_cost_one_kilometer_minimum_electricity_kilometers_l69_6930
