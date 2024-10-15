import Mathlib

namespace NUMINAMATH_GPT_shoes_ratio_l1529_152911

theorem shoes_ratio (Scott_shoes : ℕ) (m : ℕ) (h1 : Scott_shoes = 7)
  (h2 : ∀ Anthony_shoes, Anthony_shoes = m * Scott_shoes)
  (h3 : ∀ Jim_shoes, Jim_shoes = Anthony_shoes - 2)
  (h4 : ∀ Anthony_shoes Jim_shoes, Anthony_shoes = Jim_shoes + 2) : 
  ∃ m : ℕ, (Anthony_shoes / Scott_shoes) = m := 
by 
  sorry

end NUMINAMATH_GPT_shoes_ratio_l1529_152911


namespace NUMINAMATH_GPT_second_car_avg_mpg_l1529_152962

theorem second_car_avg_mpg 
  (x y : ℝ) 
  (h1 : x + y = 75) 
  (h2 : 25 * x + 35 * y = 2275) : 
  y = 40 := 
by sorry

end NUMINAMATH_GPT_second_car_avg_mpg_l1529_152962


namespace NUMINAMATH_GPT_total_students_l1529_152945

-- Definitions based on the conditions:
def yoongi_left : ℕ := 7
def yoongi_right : ℕ := 5

-- Theorem statement that proves the total number of students given the conditions
theorem total_students (y_left y_right : ℕ) : y_left = yoongi_left -> y_right = yoongi_right -> (y_left + y_right - 1) = 11 := 
by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_total_students_l1529_152945


namespace NUMINAMATH_GPT_equation_of_line_l_l1529_152984

def point (P : ℝ × ℝ) := P = (2, 1)
def parallel (x y : ℝ) : Prop := 2 * x - y + 2 = 0

theorem equation_of_line_l (c : ℝ) (x y : ℝ) :
  (parallel x y ∧ point (x, y)) →
  2 * x - y + c = 0 →
  c = -3 → 2 * x - y - 3 = 0 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_equation_of_line_l_l1529_152984


namespace NUMINAMATH_GPT_value_of_a_when_x_is_3_root_l1529_152927

theorem value_of_a_when_x_is_3_root (a : ℝ) :
  (3 ^ 2 + 3 * a + 9 = 0) -> a = -6 := by
  intros h
  sorry

end NUMINAMATH_GPT_value_of_a_when_x_is_3_root_l1529_152927


namespace NUMINAMATH_GPT_laura_owes_amount_l1529_152906

-- Define the given conditions as variables
def principal : ℝ := 35
def rate : ℝ := 0.05
def time : ℝ := 1

-- Define the interest calculation
def interest : ℝ := principal * rate * time

-- Define the final amount owed calculation
def amount_owed : ℝ := principal + interest

-- State the theorem we want to prove
theorem laura_owes_amount
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (interest : ℝ := principal * rate * time)
  (amount_owed : ℝ := principal + interest) :
  amount_owed = 36.75 := 
by 
  -- proof would go here
  sorry

end NUMINAMATH_GPT_laura_owes_amount_l1529_152906


namespace NUMINAMATH_GPT_earnings_difference_l1529_152950

-- Define the prices and quantities.
def price_A : ℝ := 4
def price_B : ℝ := 3.5
def quantity_A : ℕ := 300
def quantity_B : ℕ := 350

-- Define the earnings for both companies.
def earnings_A := price_A * quantity_A
def earnings_B := price_B * quantity_B

-- State the theorem we intend to prove.
theorem earnings_difference :
  earnings_B - earnings_A = 25 := by
  sorry

end NUMINAMATH_GPT_earnings_difference_l1529_152950


namespace NUMINAMATH_GPT_difference_even_odd_sums_l1529_152987

def sum_first_n_even_numbers (n : ℕ) : ℕ := n * (n + 1)
def sum_first_n_odd_numbers (n : ℕ) : ℕ := n * n

theorem difference_even_odd_sums : sum_first_n_even_numbers 1001 - sum_first_n_odd_numbers 1001 = 1001 := by
  sorry

end NUMINAMATH_GPT_difference_even_odd_sums_l1529_152987


namespace NUMINAMATH_GPT_distance_between_homes_l1529_152954

theorem distance_between_homes (Maxwell_speed : ℝ) (Brad_speed : ℝ) (M_time : ℝ) (B_delay : ℝ) (D : ℝ) 
  (h1 : Maxwell_speed = 4) 
  (h2 : Brad_speed = 6)
  (h3 : M_time = 8)
  (h4 : B_delay = 1) :
  D = 74 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_homes_l1529_152954


namespace NUMINAMATH_GPT_b_car_usage_hours_l1529_152932

theorem b_car_usage_hours (h : ℕ) (total_cost_a_b_c : ℕ) 
  (a_usage : ℕ) (b_payment : ℕ) (c_usage : ℕ) 
  (total_cost : total_cost_a_b_c = 720)
  (usage_a : a_usage = 9) 
  (usage_c : c_usage = 13)
  (payment_b : b_payment = 225) 
  (cost_per_hour : ℝ := total_cost_a_b_c / (a_usage + h + c_usage)) :
  b_payment = cost_per_hour * h → h = 10 := 
by
  sorry

end NUMINAMATH_GPT_b_car_usage_hours_l1529_152932


namespace NUMINAMATH_GPT_minimum_tanA_9tanB_l1529_152933

variable (a b c A B : ℝ)
variable (Aacute : A > 0 ∧ A < π / 2)
variable (h1 : a^2 = b^2 + 2*b*c * Real.sin A)
variable (habc : a = b * Real.sin A)

theorem minimum_tanA_9tanB : 
  ∃ (A B : ℝ), (A > 0 ∧ A < π / 2) ∧ (a^2 = b^2 + 2*b*c * Real.sin A) ∧ (a = b * Real.sin A) ∧ 
  (min ((Real.tan A) - 9*(Real.tan B)) = -2) := 
  sorry

end NUMINAMATH_GPT_minimum_tanA_9tanB_l1529_152933


namespace NUMINAMATH_GPT_quadratic_expression_value_l1529_152922

theorem quadratic_expression_value (a : ℝ) :
  (∃ x : ℝ, (3 * a - 1) * x^2 - a * x + 1 / 4 = 0 ∧ 
  (3 * a - 1) * x^2 - a * x + 1 / 4 = 0 ∧ 
  a^2 - 3 * a + 1 = 0) → 
  a^2 - 2 * a + 2021 + 1 / a = 2023 := 
sorry

end NUMINAMATH_GPT_quadratic_expression_value_l1529_152922


namespace NUMINAMATH_GPT_cos_double_angle_l1529_152997

theorem cos_double_angle (k : ℝ) (h : Real.sin (10 * Real.pi / 180) = k) : 
  Real.cos (20 * Real.pi / 180) = 1 - 2 * k^2 := by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1529_152997


namespace NUMINAMATH_GPT_domain_of_tan_2x_plus_pi_over_3_l1529_152930

noncomputable def domain_tan_transformed : Set ℝ :=
  {x : ℝ | ∀ (k : ℤ), x ≠ k * (Real.pi / 2) + (Real.pi / 12)}

theorem domain_of_tan_2x_plus_pi_over_3 :
  (∀ x : ℝ, x ∉ domain_tan_transformed ↔ ∃ (k : ℤ), x = k * (Real.pi / 2) + (Real.pi / 12)) :=
sorry

end NUMINAMATH_GPT_domain_of_tan_2x_plus_pi_over_3_l1529_152930


namespace NUMINAMATH_GPT_smallest_integer_in_set_A_l1529_152931

def set_A : Set ℝ := {x | |x - 2| ≤ 5}

theorem smallest_integer_in_set_A : ∃ m ∈ set_A, ∀ n ∈ set_A, m ≤ n := 
  sorry

end NUMINAMATH_GPT_smallest_integer_in_set_A_l1529_152931


namespace NUMINAMATH_GPT_probability_linda_picks_letter_in_mathematics_l1529_152960

def english_alphabet : Finset Char := "ABCDEFGHIJKLMNOPQRSTUVWXYZ".toList.toFinset

def word_mathematics : Finset Char := "MATHEMATICS".toList.toFinset

theorem probability_linda_picks_letter_in_mathematics : 
  (word_mathematics.card : ℚ) / (english_alphabet.card : ℚ) = 4 / 13 := by sorry

end NUMINAMATH_GPT_probability_linda_picks_letter_in_mathematics_l1529_152960


namespace NUMINAMATH_GPT_factory_hours_per_day_l1529_152956

def factory_produces (hours_per_day : ℕ) : Prop :=
  let refrigerators_per_hour := 90
  let coolers_per_hour := 160
  let total_products_per_hour := refrigerators_per_hour + coolers_per_hour
  let total_products_in_5_days := 11250
  total_products_per_hour * (5 * hours_per_day) = total_products_in_5_days

theorem factory_hours_per_day : ∃ h : ℕ, factory_produces h ∧ h = 9 :=
by
  existsi 9
  unfold factory_produces
  sorry

end NUMINAMATH_GPT_factory_hours_per_day_l1529_152956


namespace NUMINAMATH_GPT_cost_of_blue_pill_l1529_152948

/-
Statement:
Bob takes two blue pills and one orange pill each day for three weeks.
The cost of a blue pill is $2 more than an orange pill.
The total cost for all pills over the three weeks amounts to $966.
Prove that the cost of one blue pill is $16.
-/

theorem cost_of_blue_pill (days : ℕ) (total_cost : ℝ) (cost_orange : ℝ) (cost_blue : ℝ) 
  (h1 : days = 21) 
  (h2 : total_cost = 966) 
  (h3 : cost_blue = cost_orange + 2) 
  (daily_pill_cost : ℝ)
  (h4 : daily_pill_cost = total_cost / days)
  (h5 : daily_pill_cost = 2 * cost_blue + cost_orange) :
  cost_blue = 16 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_blue_pill_l1529_152948


namespace NUMINAMATH_GPT_num_routes_M_to_N_l1529_152973

-- Define the relevant points and connections as predicates
def can_reach_directly (x y : String) : Prop :=
  if (x = "C" ∧ y = "N") ∨ (x = "D" ∧ y = "N") ∨ (x = "B" ∧ y = "N") then true else false

def can_reach_via (x y z : String) : Prop :=
  if (x = "A" ∧ y = "C" ∧ z = "N") ∨ (x = "A" ∧ y = "D" ∧ z = "N") ∨ (x = "B" ∧ y = "A" ∧ z = "N") ∨ 
     (x = "B" ∧ y = "C" ∧ z = "N") ∨ (x = "E" ∧ y = "B" ∧ z = "N") ∨ (x = "F" ∧ y = "A" ∧ z = "N") ∨ 
     (x = "F" ∧ y = "B" ∧ z = "N") then true else false

-- Define a function to compute the number of ways from a starting point to "N"
noncomputable def num_routes_to_N : String → ℕ
| "N" => 1
| "C" => 1
| "D" => 1
| "A" => 2 -- from C to N and D to N
| "B" => 4 -- from B to N directly, from B to N via A (2 ways), from B to N via C
| "E" => 4 -- from E to N via B
| "F" => 6 -- from F to N via A (2 ways), from F to N via B (4 ways)
| "M" => 16 -- from M to N via A, B, E, F
| _ => 0

-- The theorem statement
theorem num_routes_M_to_N : num_routes_to_N "M" = 16 :=
by
  sorry

end NUMINAMATH_GPT_num_routes_M_to_N_l1529_152973


namespace NUMINAMATH_GPT_sum_of_integers_l1529_152996

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 180) : x + y = 28 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l1529_152996


namespace NUMINAMATH_GPT_dimes_turned_in_l1529_152968

theorem dimes_turned_in (total_coins nickels quarters : ℕ) (h1 : total_coins = 11) (h2 : nickels = 2) (h3 : quarters = 7) : 
  ∃ dimes : ℕ, dimes + nickels + quarters = total_coins ∧ dimes = 2 :=
by
  sorry

end NUMINAMATH_GPT_dimes_turned_in_l1529_152968


namespace NUMINAMATH_GPT_usual_time_is_42_l1529_152947

noncomputable def usual_time_to_school (R T : ℝ) := T * R
noncomputable def improved_time_to_school (R T : ℝ) := ((7/6) * R) * (T - 6)

theorem usual_time_is_42 (R T : ℝ) :
  (usual_time_to_school R T) = (improved_time_to_school R T) → T = 42 :=
by
  sorry

end NUMINAMATH_GPT_usual_time_is_42_l1529_152947


namespace NUMINAMATH_GPT_total_sheep_l1529_152974

variable (x y : ℕ)
/-- Initial condition: After one ram runs away, the ratio of rams to ewes is 7:5. -/
def initial_ratio (x y : ℕ) : Prop := 5 * (x - 1) = 7 * y
/-- Second condition: After the ram returns and one ewe runs away, the ratio of rams to ewes is 5:3. -/
def second_ratio (x y : ℕ) : Prop := 3 * x = 5 * (y - 1)
/-- The total number of sheep in the flock initially is 25. -/
theorem total_sheep (x y : ℕ) 
  (h1 : initial_ratio x y) 
  (h2 : second_ratio x y) : 
  x + y = 25 := 
by sorry

end NUMINAMATH_GPT_total_sheep_l1529_152974


namespace NUMINAMATH_GPT_remainder_xyz_mod7_condition_l1529_152965

-- Define variables and conditions
variables (x y z : ℕ)
theorem remainder_xyz_mod7_condition (hx : x < 7) (hy : y < 7) (hz : z < 7)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 7])
  (h2 : 3 * x + 2 * y + z ≡ 2 [MOD 7])
  (h3 : 2 * x + y + 3 * z ≡ 3 [MOD 7]) :
  (x * y * z % 7) ≡ 1 [MOD 7] := sorry

end NUMINAMATH_GPT_remainder_xyz_mod7_condition_l1529_152965


namespace NUMINAMATH_GPT_lila_substituted_value_l1529_152966

theorem lila_substituted_value:
  let a := 2
  let b := 3
  let c := 4
  let d := 5
  let f := 6
  ∃ e : ℚ, 20 * e = 2 * (3 - 4 * (5 - (e / 6))) ∧ e = -51 / 28 := sorry

end NUMINAMATH_GPT_lila_substituted_value_l1529_152966


namespace NUMINAMATH_GPT_lower_bound_expression_l1529_152969

theorem lower_bound_expression (n : ℤ) (L : ℤ) :
  (∃ k : ℕ, k = 20 ∧
          ∀ n, (L < 4 * n + 7 ∧ 4 * n + 7 < 80)) →
  L = 3 :=
by
  sorry

end NUMINAMATH_GPT_lower_bound_expression_l1529_152969


namespace NUMINAMATH_GPT_leaves_blew_away_correct_l1529_152980

-- Define the initial number of leaves Mikey had.
def initial_leaves : ℕ := 356

-- Define the number of leaves Mikey has left.
def leaves_left : ℕ := 112

-- Define the number of leaves that blew away.
def leaves_blew_away : ℕ := initial_leaves - leaves_left

-- Prove that the number of leaves that blew away is 244.
theorem leaves_blew_away_correct : leaves_blew_away = 244 :=
by sorry

end NUMINAMATH_GPT_leaves_blew_away_correct_l1529_152980


namespace NUMINAMATH_GPT_money_last_weeks_l1529_152999

theorem money_last_weeks (mowing_earning : ℕ) (weeding_earning : ℕ) (spending_per_week : ℕ) 
  (total_amount : ℕ) (weeks : ℕ) :
  mowing_earning = 9 →
  weeding_earning = 18 →
  spending_per_week = 3 →
  total_amount = mowing_earning + weeding_earning →
  weeks = total_amount / spending_per_week →
  weeks = 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_money_last_weeks_l1529_152999


namespace NUMINAMATH_GPT_min_value_expression_l1529_152964

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 
  (1 / a + 2) * (1 / b + 2) ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1529_152964


namespace NUMINAMATH_GPT_smallest_solution_l1529_152971

def polynomial (x : ℝ) := x^4 - 34 * x^2 + 225 = 0

theorem smallest_solution : ∃ x : ℝ, polynomial x ∧ ∀ y : ℝ, polynomial y → x ≤ y := 
sorry

end NUMINAMATH_GPT_smallest_solution_l1529_152971


namespace NUMINAMATH_GPT_count_valid_three_digit_numbers_l1529_152928

def three_digit_number (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a * 100 + b * 10 + c < 1000) ∧
  (a * 100 + b * 10 + c >= 100) ∧
  (c = 2 * (b - a) + a)

theorem count_valid_three_digit_numbers : ∃ n : ℕ, n = 90 ∧
  ∃ (a b c : ℕ), three_digit_number a b c :=
by
  sorry

end NUMINAMATH_GPT_count_valid_three_digit_numbers_l1529_152928


namespace NUMINAMATH_GPT_inequality_proof_l1529_152970

theorem inequality_proof 
  (x y z w : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w)
  (h_eq : (x^3 + y^3)^4 = z^3 + w^3) :
  x^4 * z + y^4 * w ≥ z * w :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1529_152970


namespace NUMINAMATH_GPT_circle_area_from_diameter_points_l1529_152958

theorem circle_area_from_diameter_points (C D : ℝ × ℝ)
    (hC : C = (-2, 3)) (hD : D = (4, -1)) :
    ∃ (A : ℝ), A = 13 * Real.pi :=
by
  let distance := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  have diameter : distance = Real.sqrt (6^2 + (-4)^2) := sorry -- this follows from the coordinates
  have radius : distance / 2 = Real.sqrt 13 := sorry -- half of the diameter
  exact ⟨13 * Real.pi, sorry⟩ -- area of the circle

end NUMINAMATH_GPT_circle_area_from_diameter_points_l1529_152958


namespace NUMINAMATH_GPT_quadratic_inequality_empty_set_l1529_152953

theorem quadratic_inequality_empty_set (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 - ax + 1 < 0)) ↔ (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_empty_set_l1529_152953


namespace NUMINAMATH_GPT_find_second_number_l1529_152900

theorem find_second_number :
  ∃ (x y : ℕ), (y = x + 4) ∧ (x + y = 56) ∧ (y = 30) :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l1529_152900


namespace NUMINAMATH_GPT_problem_l1529_152917

-- Definition of triangular number
def is_triangular (n k : ℕ) := n = k * (k + 1) / 2

-- Definition of choosing 2 marbles
def choose_2 (n m : ℕ) := n = m * (m - 1) / 2

-- Definition of Cathy's condition
def cathy_condition (n s : ℕ) := s * s < 2 * n ∧ 2 * n - s * s = 20

theorem problem (n k m s : ℕ) :
  is_triangular n k →
  choose_2 n m →
  cathy_condition n s →
  n = 210 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1529_152917


namespace NUMINAMATH_GPT_reaction_spontaneous_at_high_temperature_l1529_152967

theorem reaction_spontaneous_at_high_temperature
  (ΔH : ℝ) (ΔS : ℝ) (T : ℝ) (ΔG : ℝ)
  (h_ΔH_pos : ΔH > 0)
  (h_ΔS_pos : ΔS > 0)
  (h_ΔG_eq : ΔG = ΔH - T * ΔS) :
  (∃ T_high : ℝ, T_high > 0 ∧ ΔG < 0) := sorry

end NUMINAMATH_GPT_reaction_spontaneous_at_high_temperature_l1529_152967


namespace NUMINAMATH_GPT_inequality_geq_l1529_152915

variable {a b c : ℝ}

theorem inequality_geq (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) := 
sorry

end NUMINAMATH_GPT_inequality_geq_l1529_152915


namespace NUMINAMATH_GPT_nada_house_size_l1529_152925

variable (N : ℕ) -- N represents the size of Nada's house

theorem nada_house_size :
  (1000 = 2 * N + 100) → (N = 450) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_nada_house_size_l1529_152925


namespace NUMINAMATH_GPT_sample_size_l1529_152920

theorem sample_size (n : ℕ) (h1 : n ∣ 36) (h2 : 36 / n ∣ 6) (h3 : (n + 1) ∣ 35) : n = 6 := 
sorry

end NUMINAMATH_GPT_sample_size_l1529_152920


namespace NUMINAMATH_GPT_max_points_per_player_l1529_152918

theorem max_points_per_player
  (num_players : ℕ)
  (total_points : ℕ)
  (min_points_per_player : ℕ)
  (extra_points : ℕ)
  (scores_by_two_or_three : Prop)
  (fouls : Prop) :
  num_players = 12 →
  total_points = 100 →
  min_points_per_player = 8 →
  scores_by_two_or_three →
  fouls →
  extra_points = (total_points - num_players * min_points_per_player) →
  q = min_points_per_player + extra_points →
  q = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_max_points_per_player_l1529_152918


namespace NUMINAMATH_GPT_brick_wall_problem_l1529_152902

theorem brick_wall_problem
  (b : ℕ)
  (rate_ben rate_arya : ℕ → ℕ)
  (combined_rate : ℕ → ℕ → ℕ)
  (work_duration : ℕ)
  (effective_combined_rate : ℕ → ℕ × ℕ → ℕ)
  (rate_ben_def : ∀ (b : ℕ), rate_ben b = b / 12)
  (rate_arya_def : ∀ (b : ℕ), rate_arya b = b / 15)
  (combined_rate_def : ∀ (b : ℕ), combined_rate (rate_ben b) (rate_arya b) = rate_ben b + rate_arya b)
  (effective_combined_rate_def : ∀ (b : ℕ), effective_combined_rate b (rate_ben b, rate_arya b) = combined_rate (rate_ben b) (rate_arya b) - 15)
  (work_duration_def : work_duration = 6)
  (completion_condition : ∀ (b : ℕ), work_duration * effective_combined_rate b (rate_ben b, rate_arya b) = b) :
  b = 900 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_brick_wall_problem_l1529_152902


namespace NUMINAMATH_GPT_eggs_needed_per_month_l1529_152921

def weekly_eggs_needed : ℕ := 10 + 14 + (14 / 2)

def weeks_in_month : ℕ := 4

def monthly_eggs_needed (weekly_eggs : ℕ) (weeks : ℕ) : ℕ :=
  weekly_eggs * weeks

theorem eggs_needed_per_month : 
  monthly_eggs_needed weekly_eggs_needed weeks_in_month = 124 :=
by {
  -- calculation details go here, but we leave it as sorry
  sorry
}

end NUMINAMATH_GPT_eggs_needed_per_month_l1529_152921


namespace NUMINAMATH_GPT_condition1_condition2_l1529_152952

-- Definition for the coordinates of point P based on given m
def P (m : ℝ) : ℝ × ℝ := (3 * m - 6, m + 1)

-- Condition 1: Point P lies on the x-axis
theorem condition1 (m : ℝ) (hx : P m = (3 * m - 6, 0)) : P m = (-9, 0) := 
by {
  -- Show that if y-coordinate is zero, then m + 1 = 0, hence m = -1
  sorry
}

-- Condition 2: Point A is (-1, 2) and AP is parallel to the y-axis
theorem condition2 (m : ℝ) (A : ℝ × ℝ := (-1, 2)) (hy : (3 * m - 6 = -1)) : P m = (-1, 8/3) :=
by {
  -- Show that if the x-coordinates of A and P are equal, then 3m-6 = -1, hence m = 5/3
  sorry
}

end NUMINAMATH_GPT_condition1_condition2_l1529_152952


namespace NUMINAMATH_GPT_problem_statement_l1529_152923

-- Mathematical Conditions
variables (a : ℝ)

-- Sufficient but not necessary condition proof statement
def sufficient_but_not_necessary : Prop :=
  (∀ a : ℝ, a > 0 → a^2 + a ≥ 0) ∧ ¬(∀ a : ℝ, a^2 + a ≥ 0 → a > 0)

-- Main problem to be proved
theorem problem_statement : sufficient_but_not_necessary :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1529_152923


namespace NUMINAMATH_GPT_jane_albert_same_committee_l1529_152936

def probability_same_committee (total_MBAs : ℕ) (committee_size : ℕ) (num_committees : ℕ) (favorable_cases : ℕ) (total_cases : ℕ) : ℚ :=
  favorable_cases / total_cases

theorem jane_albert_same_committee :
  probability_same_committee 9 4 3 105 630 = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_jane_albert_same_committee_l1529_152936


namespace NUMINAMATH_GPT_largest_prime_divisor_l1529_152913

theorem largest_prime_divisor : ∃ p : ℕ, Nat.Prime p ∧ p ∣ (17^2 + 60^2) ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (17^2 + 60^2) → q ≤ p :=
  sorry

end NUMINAMATH_GPT_largest_prime_divisor_l1529_152913


namespace NUMINAMATH_GPT_probability_heads_l1529_152972

theorem probability_heads (p : ℝ) (q : ℝ) (C_10_5 C_10_6 : ℝ)
  (h1 : q = 1 - p)
  (h2 : C_10_5 = 252)
  (h3 : C_10_6 = 210)
  (eqn : C_10_5 * p^5 * q^5 = C_10_6 * p^6 * q^4) :
  p = 6 / 11 := 
by
  sorry

end NUMINAMATH_GPT_probability_heads_l1529_152972


namespace NUMINAMATH_GPT_solve_problem_statement_l1529_152963

def problem_statement : Prop :=
  ∃ n, 3^19 % n = 7 ∧ n = 1162261460

theorem solve_problem_statement : problem_statement :=
  sorry

end NUMINAMATH_GPT_solve_problem_statement_l1529_152963


namespace NUMINAMATH_GPT_sqrt_div_value_l1529_152942

open Real

theorem sqrt_div_value (n x : ℝ) (h1 : n = 3600) (h2 : sqrt n / x = 4) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_div_value_l1529_152942


namespace NUMINAMATH_GPT_voldemort_calorie_intake_limit_l1529_152994

theorem voldemort_calorie_intake_limit :
  let breakfast := 560
  let lunch := 780
  let cake := 110
  let chips := 310
  let coke := 215
  let dinner := cake + chips + coke
  let remaining := 525
  breakfast + lunch + dinner + remaining = 2500 :=
by
  -- to clarify, the statement alone is provided, so we add 'sorry' to omit the actual proof steps
  sorry

end NUMINAMATH_GPT_voldemort_calorie_intake_limit_l1529_152994


namespace NUMINAMATH_GPT_kay_age_l1529_152903

/-- Let K be Kay's age. If the youngest sibling is 5 less 
than half of Kay's age, the oldest sibling is four times 
as old as the youngest sibling, and the oldest sibling 
is 44 years old, then Kay is 32 years old. -/
theorem kay_age (K : ℕ) (youngest oldest : ℕ) 
  (h1 : youngest = (K / 2) - 5)
  (h2 : oldest = 4 * youngest)
  (h3 : oldest = 44) : K = 32 := 
by
  sorry

end NUMINAMATH_GPT_kay_age_l1529_152903


namespace NUMINAMATH_GPT_root_in_interval_l1529_152905

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 3

theorem root_in_interval : ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 :=
  sorry

end NUMINAMATH_GPT_root_in_interval_l1529_152905


namespace NUMINAMATH_GPT_range_of_c_l1529_152985

theorem range_of_c (a c : ℝ) (ha : a ≥ 1 / 8)
  (h : ∀ x > 0, 2 * x + a / x ≥ c) : c ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_c_l1529_152985


namespace NUMINAMATH_GPT_SarahsNumber_is_2880_l1529_152957

def SarahsNumber (n : ℕ) : Prop :=
  (144 ∣ n) ∧ (45 ∣ n) ∧ (1000 ≤ n ∧ n ≤ 3000)

theorem SarahsNumber_is_2880 : SarahsNumber 2880 :=
  by
  sorry

end NUMINAMATH_GPT_SarahsNumber_is_2880_l1529_152957


namespace NUMINAMATH_GPT_cookies_per_sheet_is_16_l1529_152901

-- Define the number of members
def members : ℕ := 100

-- Define the number of sheets each member bakes
def sheets_per_member : ℕ := 10

-- Define the total number of cookies baked
def total_cookies : ℕ := 16000

-- Calculate the total number of sheets baked
def total_sheets : ℕ := members * sheets_per_member

-- Define the number of cookies per sheet as a result of given conditions
def cookies_per_sheet : ℕ := total_cookies / total_sheets

-- Prove that the number of cookies on each sheet is 16 given the conditions
theorem cookies_per_sheet_is_16 : cookies_per_sheet = 16 :=
by
  -- Assuming all the given definitions and conditions
  sorry

end NUMINAMATH_GPT_cookies_per_sheet_is_16_l1529_152901


namespace NUMINAMATH_GPT_dartboard_distribution_count_l1529_152939

-- Definition of the problem in Lean 4
def count_dartboard_distributions : ℕ :=
  -- We directly use the identified correct answer
  5

theorem dartboard_distribution_count :
  count_dartboard_distributions = 5 :=
sorry

end NUMINAMATH_GPT_dartboard_distribution_count_l1529_152939


namespace NUMINAMATH_GPT_ways_to_seat_people_l1529_152941

noncomputable def number_of_ways : ℕ :=
  let choose_people := (Nat.choose 12 8)
  let divide_groups := (Nat.choose 8 4)
  let arrange_circular_table := (Nat.factorial 3)
  choose_people * divide_groups * (arrange_circular_table * arrange_circular_table)

theorem ways_to_seat_people :
  number_of_ways = 1247400 :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_ways_to_seat_people_l1529_152941


namespace NUMINAMATH_GPT_product_of_numbers_l1529_152943

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 21) (h2 : x^2 + y^2 = 527) : x * y = -43.05 := by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l1529_152943


namespace NUMINAMATH_GPT_people_per_entrance_l1529_152975

theorem people_per_entrance (e p : ℕ) (h1 : e = 5) (h2 : p = 1415) : p / e = 283 := by
  sorry

end NUMINAMATH_GPT_people_per_entrance_l1529_152975


namespace NUMINAMATH_GPT_number_of_female_students_l1529_152977

theorem number_of_female_students (M F : ℕ) (h1 : F = M + 6) (h2 : M + F = 82) : F = 44 :=
by
  sorry

end NUMINAMATH_GPT_number_of_female_students_l1529_152977


namespace NUMINAMATH_GPT_inequality_solution_l1529_152949

theorem inequality_solution {x : ℝ} :
  (12 * x^2 + 24 * x - 75) / ((3 * x - 5) * (x + 5)) < 4 ↔ -5 < x ∧ x < 5 / 3 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1529_152949


namespace NUMINAMATH_GPT_min_value_of_expression_l1529_152919

theorem min_value_of_expression
  (x y : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (h_eq : x * (x + y) = 5 * x + y) : 2 * x + y ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1529_152919


namespace NUMINAMATH_GPT_frank_used_2_bags_l1529_152983

theorem frank_used_2_bags (total_candy : ℕ) (candy_per_bag : ℕ) (h1 : total_candy = 16) (h2 : candy_per_bag = 8) : (total_candy / candy_per_bag) = 2 := 
by
  sorry

end NUMINAMATH_GPT_frank_used_2_bags_l1529_152983


namespace NUMINAMATH_GPT_interest_paid_percent_l1529_152986

noncomputable def down_payment : ℝ := 300
noncomputable def total_cost : ℝ := 750
noncomputable def monthly_payment : ℝ := 57
noncomputable def final_payment : ℝ := 21
noncomputable def num_monthly_payments : ℕ := 9

noncomputable def total_instalments : ℝ := (num_monthly_payments * monthly_payment) + final_payment
noncomputable def total_paid : ℝ := total_instalments + down_payment
noncomputable def amount_borrowed : ℝ := total_cost - down_payment
noncomputable def interest_paid : ℝ := total_paid - amount_borrowed
noncomputable def interest_percent : ℝ := (interest_paid / amount_borrowed) * 100

theorem interest_paid_percent:
  interest_percent = 85.33 := by
  sorry

end NUMINAMATH_GPT_interest_paid_percent_l1529_152986


namespace NUMINAMATH_GPT_coordinates_of_A_l1529_152998

-- Define initial coordinates of point A
def A : ℝ × ℝ := (-2, 4)

-- Define the transformation of moving 2 units upwards
def move_up (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1, point.2 + units)

-- Define the transformation of moving 3 units to the left
def move_left (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1 - units, point.2)

-- Combine the transformations to get point A'
def A' : ℝ × ℝ :=
  move_left (move_up A 2) 3

-- The theorem stating that A' is (-5, 6)
theorem coordinates_of_A' : A' = (-5, 6) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_A_l1529_152998


namespace NUMINAMATH_GPT_evaluate_expression_l1529_152990

variable (y : ℕ)

theorem evaluate_expression (h : y = 3) : 
    (y^(1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19) / y^(2 + 4 + 6 + 8 + 10 + 12)) = 3^58 :=
by
  -- Proof will be done here
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1529_152990


namespace NUMINAMATH_GPT_sqrt_180_eq_l1529_152982

noncomputable def simplify_sqrt_180 : Real := 6 * Real.sqrt 5

theorem sqrt_180_eq : Real.sqrt 180 = simplify_sqrt_180 := 
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_sqrt_180_eq_l1529_152982


namespace NUMINAMATH_GPT_kendall_total_distance_l1529_152946

def distance_with_mother : ℝ := 0.17
def distance_with_father : ℝ := 0.5
def total_distance : ℝ := 0.67

theorem kendall_total_distance :
  (distance_with_mother + distance_with_father = total_distance) :=
sorry

end NUMINAMATH_GPT_kendall_total_distance_l1529_152946


namespace NUMINAMATH_GPT_greatest_b_value_ineq_l1529_152961

theorem greatest_b_value_ineq (b : ℝ) (h : -b^2 + 8 * b - 15 ≥ 0) : b ≤ 5 := 
sorry

end NUMINAMATH_GPT_greatest_b_value_ineq_l1529_152961


namespace NUMINAMATH_GPT_find_x_l1529_152944

theorem find_x : ∃ x : ℝ, (3 * (x + 2 - 6)) / 4 = 3 ∧ x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1529_152944


namespace NUMINAMATH_GPT_sin1993_cos1993_leq_zero_l1529_152938

theorem sin1993_cos1993_leq_zero (x : ℝ) (h : Real.sin x + Real.cos x ≤ 0) : 
  Real.sin x ^ 1993 + Real.cos x ^ 1993 ≤ 0 := 
by 
  sorry

end NUMINAMATH_GPT_sin1993_cos1993_leq_zero_l1529_152938


namespace NUMINAMATH_GPT_probability_A_does_not_lose_l1529_152909

theorem probability_A_does_not_lose (pA_wins p_draw : ℝ) (hA_wins : pA_wins = 0.4) (h_draw : p_draw = 0.2) :
  pA_wins + p_draw = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_probability_A_does_not_lose_l1529_152909


namespace NUMINAMATH_GPT_largest_four_digit_negative_congruent_3_mod_29_l1529_152976

theorem largest_four_digit_negative_congruent_3_mod_29 : 
  ∃ (n : ℤ), n < 0 ∧ n ≥ -9999 ∧ (n % 29 = 3) ∧ n = -1012 :=
sorry

end NUMINAMATH_GPT_largest_four_digit_negative_congruent_3_mod_29_l1529_152976


namespace NUMINAMATH_GPT_tangent_point_condition_l1529_152959

open Function

def f (x : ℝ) : ℝ := x^3 - 3 * x
def tangent_line (s : ℝ) (x t : ℝ) : ℝ := (3 * s^2 - 3) * (x - 2) + s^3 - 3 * s

theorem tangent_point_condition (t : ℝ) (h_tangent : ∃s : ℝ, tangent_line s 2 t = t) 
  (h_not_on_curve : ∀ s, (2, t) ≠ (s, f s)) : t = -6 :=
by
  sorry

end NUMINAMATH_GPT_tangent_point_condition_l1529_152959


namespace NUMINAMATH_GPT_quadratic_no_real_roots_range_l1529_152907

theorem quadratic_no_real_roots_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) → k < -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_range_l1529_152907


namespace NUMINAMATH_GPT_find_number_l1529_152988

theorem find_number (x : ℕ) (h : x * 12 = 540) : x = 45 :=
by sorry

end NUMINAMATH_GPT_find_number_l1529_152988


namespace NUMINAMATH_GPT_children_got_on_bus_l1529_152912

-- Definitions based on conditions
def initial_children : ℕ := 22
def children_got_off : ℕ := 60
def children_after_stop : ℕ := 2

-- Define the problem
theorem children_got_on_bus : ∃ x : ℕ, initial_children - children_got_off + x = children_after_stop ∧ x = 40 :=
by
  sorry

end NUMINAMATH_GPT_children_got_on_bus_l1529_152912


namespace NUMINAMATH_GPT_at_least_six_destinations_l1529_152908

theorem at_least_six_destinations (destinations : ℕ) (tickets_sold : ℕ) (h_dest : destinations = 200) (h_tickets : tickets_sold = 3800) :
  ∃ k ≥ 6, ∃ t : ℕ, (∃ f : Fin destinations → ℕ, (∀ i : Fin destinations, f i ≤ t) ∧ (tickets_sold ≤ t * destinations) ∧ ((∃ i : Fin destinations, f i = k) → k ≥ 6)) :=
by
  sorry

end NUMINAMATH_GPT_at_least_six_destinations_l1529_152908


namespace NUMINAMATH_GPT_each_episode_length_l1529_152937

theorem each_episode_length (h_watch_time : ∀ d : ℕ, d = 5 → 2 * 60 * d = 600)
  (h_episodes : 20 > 0) : 600 / 20 = 30 := by
  -- Conditions used:
  -- 1. h_watch_time : John wants to finish a show in 5 days by watching 2 hours a day.
  -- 2. h_episodes : There are 20 episodes.
  -- Goal: Prove that each episode is 30 minutes long.
  sorry

end NUMINAMATH_GPT_each_episode_length_l1529_152937


namespace NUMINAMATH_GPT_largest_fully_communicating_sets_eq_l1529_152992

noncomputable def largest_fully_communicating_sets :=
  let total_sets := Nat.choose 99 4
  let non_communicating_sets_per_pod := Nat.choose 48 3
  let total_non_communicating_sets := 99 * non_communicating_sets_per_pod
  total_sets - total_non_communicating_sets

theorem largest_fully_communicating_sets_eq : largest_fully_communicating_sets = 2051652 := by
  sorry

end NUMINAMATH_GPT_largest_fully_communicating_sets_eq_l1529_152992


namespace NUMINAMATH_GPT_train_speed_correct_l1529_152981

theorem train_speed_correct :
  ∀ (L : ℝ) (V_man : ℝ) (T : ℝ) (V_train : ℝ),
    L = 220 ∧ V_man = 6 * (1000 / 3600) ∧ T = 11.999040076793857 ∧ 
    L / T - V_man = V_train ↔ V_train * 3.6 = 60 :=
by
  intros L V_man T V_train
  sorry

end NUMINAMATH_GPT_train_speed_correct_l1529_152981


namespace NUMINAMATH_GPT_probability_of_exactly_one_second_class_product_l1529_152914

-- Definitions based on the conditions provided
def total_products := 100
def first_class_products := 90
def second_class_products := 10
def selected_products := 4

-- Calculation of the probability
noncomputable def probability : ℚ :=
  (Nat.choose 10 1 * Nat.choose 90 3) / Nat.choose 100 4

-- Statement to prove that the probability is 0.30
theorem probability_of_exactly_one_second_class_product : 
  probability = 0.30 := by
  sorry

end NUMINAMATH_GPT_probability_of_exactly_one_second_class_product_l1529_152914


namespace NUMINAMATH_GPT_solution_set_inequality_l1529_152916

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) / (x + 2) < 0 ↔ -2 < x ∧ x < 3 :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1529_152916


namespace NUMINAMATH_GPT_ways_to_choose_providers_l1529_152940

theorem ways_to_choose_providers : (25 * 24 * 23 * 22 = 303600) :=
by
  sorry

end NUMINAMATH_GPT_ways_to_choose_providers_l1529_152940


namespace NUMINAMATH_GPT_linear_function_passing_through_point_and_intersecting_another_line_area_of_triangle_l1529_152934

theorem linear_function_passing_through_point_and_intersecting_another_line (
  k b : ℝ)
  (h1 : (∀ x y : ℝ, y = k * x + b → ((x = 3 ∧ y = -3) ∨ (x = 3/4 ∧ y = 0))))
  (h2 : (∀ x : ℝ, 0 = (4 * x - 3) → x = 3/4))
  : k = -4 / 3 ∧ b = 1 := 
sorry

theorem area_of_triangle (
  k b : ℝ)
  (h1 : k = -4 / 3 ∧ b = 1)
  : 1 / 2 * 3 / 4 * 1 = 3 / 8 := 
sorry

end NUMINAMATH_GPT_linear_function_passing_through_point_and_intersecting_another_line_area_of_triangle_l1529_152934


namespace NUMINAMATH_GPT_find_circle_equation_l1529_152926

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

-- Define the equation of the asymptote
def asymptote (x y : ℝ) : Prop :=
  4 * x - 3 * y = 0

-- Define the given center of the circle
def center : ℝ × ℝ :=
  (5, 0)

-- Define the radius of the circle
def radius : ℝ :=
  4

-- Define the circle in center-radius form and expand it to standard form
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 * x + 9 = 0

theorem find_circle_equation 
  (x y : ℝ) 
  (h : asymptote x y)
  (h_center : (x, y) = center) 
  (h_radius : radius = 4) : circle_eq x y :=
sorry

end NUMINAMATH_GPT_find_circle_equation_l1529_152926


namespace NUMINAMATH_GPT_eval_expression_l1529_152989

theorem eval_expression (m : ℝ) (h : m^2 + 2*m - 1 = 0) : 2*m^2 + 4*m - 3 = -1 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1529_152989


namespace NUMINAMATH_GPT_min_value_of_f_l1529_152904

def f (x : ℝ) := abs (x + 1) + abs (x + 3) + abs (x + 6)

theorem min_value_of_f : ∃ (x : ℝ), f x = 5 :=
by
  use -3
  simp [f]
  sorry

end NUMINAMATH_GPT_min_value_of_f_l1529_152904


namespace NUMINAMATH_GPT_bigger_number_in_ratio_l1529_152978

theorem bigger_number_in_ratio (x : ℕ) (h : 11 * x = 143) : 8 * x = 104 :=
by
  sorry

end NUMINAMATH_GPT_bigger_number_in_ratio_l1529_152978


namespace NUMINAMATH_GPT_molecular_weight_CaH2_correct_l1529_152929

-- Define the atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_H : ℝ := 1.008

-- Define the formula to compute the molecular weight
def molecular_weight_CaH2 (atomic_weight_Ca : ℝ) (atomic_weight_H : ℝ) : ℝ :=
  (1 * atomic_weight_Ca) + (2 * atomic_weight_H)

-- Theorem stating that the molecular weight of CaH2 is 42.096 g/mol
theorem molecular_weight_CaH2_correct : molecular_weight_CaH2 atomic_weight_Ca atomic_weight_H = 42.096 := 
by 
  sorry

end NUMINAMATH_GPT_molecular_weight_CaH2_correct_l1529_152929


namespace NUMINAMATH_GPT_number_of_dimes_l1529_152993

theorem number_of_dimes (d q : ℕ) (h₁ : 10 * d + 25 * q = 580) (h₂ : d = q + 10) : d = 23 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_dimes_l1529_152993


namespace NUMINAMATH_GPT_no_integer_solution_l1529_152910

theorem no_integer_solution (m n : ℤ) : m^2 - 11 * m * n - 8 * n^2 ≠ 88 :=
sorry

end NUMINAMATH_GPT_no_integer_solution_l1529_152910


namespace NUMINAMATH_GPT_square_circle_radius_l1529_152979

theorem square_circle_radius (a R : ℝ) (h1 : a^2 = 256) (h2 : R = 10) : R = 10 :=
sorry

end NUMINAMATH_GPT_square_circle_radius_l1529_152979


namespace NUMINAMATH_GPT_dog_food_amount_l1529_152935

theorem dog_food_amount (x : ℕ) (h1 : 3 * x + 6 = 15) : x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_dog_food_amount_l1529_152935


namespace NUMINAMATH_GPT_find_y_l1529_152995

theorem find_y (y : ℚ) (h : 6 * y + 3 * y + 4 * y + 2 * y + 1 * y + 5 * y = 360) : y = 120 / 7 := 
sorry

end NUMINAMATH_GPT_find_y_l1529_152995


namespace NUMINAMATH_GPT_tangent_line_through_point_l1529_152951

-- Definitions based purely on the conditions given in the problem.
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 25
def point_on_line (x y : ℝ) : Prop := 3 * x - 4 * y + 25 = 0
def point_given : ℝ × ℝ := (-3, 4)

-- The theorem statement to be proven
theorem tangent_line_through_point : point_on_line point_given.1 point_given.2 := 
sorry

end NUMINAMATH_GPT_tangent_line_through_point_l1529_152951


namespace NUMINAMATH_GPT_walkway_and_border_area_correct_l1529_152955

-- Definitions based on the given conditions
def flower_bed_width : ℕ := 8
def flower_bed_height : ℕ := 3
def walkway_width : ℕ := 2
def border_width : ℕ := 4
def num_rows : ℕ := 4
def num_columns : ℕ := 3

-- Total width calculation
def total_width : ℕ := 
  (flower_bed_width * num_columns) + (walkway_width * (num_columns + 1)) + (border_width * 2)

-- Total height calculation
def total_height : ℕ := 
  (flower_bed_height * num_rows) + (walkway_width * (num_rows + 1)) + (border_width * 2)

-- Total area of the garden including walkways and decorative border
def total_area : ℕ := total_width * total_height

-- Total area of flower beds
def flower_bed_area : ℕ := 
  (flower_bed_width * flower_bed_height) * (num_rows * num_columns)

-- Area of the walkways and decorative border
def walkway_and_border_area : ℕ := total_area - flower_bed_area

theorem walkway_and_border_area_correct : 
  walkway_and_border_area = 912 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_walkway_and_border_area_correct_l1529_152955


namespace NUMINAMATH_GPT_conversion_problem_l1529_152924

noncomputable def conversion1 : ℚ :=
  35 * (1/1000)  -- to convert cubic decimeters to cubic meters

noncomputable def conversion2 : ℚ :=
  53 * (1/60)  -- to convert seconds to minutes

noncomputable def conversion3 : ℚ :=
  5 * (1/60)  -- to convert minutes to hours

noncomputable def conversion4 : ℚ :=
  1 * (1/100)  -- to convert square centimeters to square decimeters

noncomputable def conversion5 : ℚ :=
  450 * (1/1000)  -- to convert milliliters to liters

theorem conversion_problem : 
  (conversion1 = 7 / 200) ∧ 
  (conversion2 = 53 / 60) ∧ 
  (conversion3 = 1 / 12) ∧ 
  (conversion4 = 1 / 100) ∧ 
  (conversion5 = 9 / 20) :=
by
  sorry

end NUMINAMATH_GPT_conversion_problem_l1529_152924


namespace NUMINAMATH_GPT_ellipse_eccentricity_m_l1529_152991

theorem ellipse_eccentricity_m (m : ℝ) (e : ℝ) (h1 : ∀ x y : ℝ, x^2 / m + y^2 = 1) (h2 : e = Real.sqrt 3 / 2) :
  m = 4 ∨ m = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_ellipse_eccentricity_m_l1529_152991
