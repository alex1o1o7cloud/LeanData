import Mathlib

namespace geom_series_common_ratio_l190_190464

theorem geom_series_common_ratio (a r S : ℝ) (h1 : S = a / (1 - r)) 
  (h2 : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
sorry

end geom_series_common_ratio_l190_190464


namespace f_div_36_l190_190929

open Nat

def f (n : ℕ) : ℕ :=
  (2 * n + 7) * 3^n + 9

theorem f_div_36 (n : ℕ) : (f n) % 36 = 0 := 
  sorry

end f_div_36_l190_190929


namespace range_of_x_l190_190779

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) :
    Real.pi / 4 ≤ x ∧ x ≤ 5 * Real.pi / 4 :=
by
  sorry

end range_of_x_l190_190779


namespace correct_calculation_l190_190671

variable (a : ℕ)

theorem correct_calculation : 
  ¬(a + a = a^2) ∧ ¬(a^3 * a = a^3) ∧ ¬(a^8 / a^2 = a^4) ∧ ((a^3)^2 = a^6) := 
by
  sorry

end correct_calculation_l190_190671


namespace factor_81_minus_27_x_cubed_l190_190262

theorem factor_81_minus_27_x_cubed (x : ℝ) : 
  81 - 27 * x ^ 3 = 27 * (3 - x) * (9 + 3 * x + x ^ 2) :=
by sorry

end factor_81_minus_27_x_cubed_l190_190262


namespace average_salary_of_employees_l190_190350

theorem average_salary_of_employees (A : ℝ)
  (h1 : 24 * A + 11500 = 25 * (A + 400)) :
  A = 1500 := 
by
  sorry

end average_salary_of_employees_l190_190350


namespace linear_function_intersects_x_axis_at_two_units_l190_190308

theorem linear_function_intersects_x_axis_at_two_units (k : ℝ) :
  (∃ x : ℝ, y = k * x + 2 ∧ y = 0 ∧ |x| = 2) ↔ k = 1 ∨ k = -1 :=
by
  sorry

end linear_function_intersects_x_axis_at_two_units_l190_190308


namespace right_triangle_legs_solutions_l190_190354

theorem right_triangle_legs_solutions (R r : ℝ) (h_cond : R / r ≥ 1 + Real.sqrt 2) :
  ∃ (a b : ℝ), 
    a = r + R + Real.sqrt (R^2 - 2 * r * R - r^2) ∧ 
    b = r + R - Real.sqrt (R^2 - 2 * r * R - r^2) ∧ 
    (2 * R)^2 = a^2 + b^2 := by
  sorry

end right_triangle_legs_solutions_l190_190354


namespace is_not_prime_390629_l190_190822

theorem is_not_prime_390629 : ¬ Prime 390629 :=
sorry

end is_not_prime_390629_l190_190822


namespace log_expression_zero_l190_190067

theorem log_expression_zero (log : Real → Real) (exp : Real → Real) (log_mul : ∀ a b, log (a * b) = log a + log b) :
  log 2 ^ 2 + log 2 * log 50 - log 4 = 0 :=
by
  sorry

end log_expression_zero_l190_190067


namespace MarionBikeCost_l190_190763

theorem MarionBikeCost (M : ℤ) (h1 : 2 * M + M = 1068) : M = 356 :=
by
  sorry

end MarionBikeCost_l190_190763


namespace energy_consumption_correct_l190_190849

def initial_wattages : List ℕ := [60, 80, 100, 120]

def increased_wattages : List ℕ := initial_wattages.map (λ x => x + (x * 25 / 100))

def combined_wattage (ws : List ℕ) : ℕ := ws.sum

def daily_energy_consumption (cw : ℕ) : ℕ := cw * 6 / 1000

def total_energy_consumption (dec : ℕ) : ℕ := dec * 30

-- Main theorem statement
theorem energy_consumption_correct :
  total_energy_consumption (daily_energy_consumption (combined_wattage increased_wattages)) = 81 := 
sorry

end energy_consumption_correct_l190_190849


namespace history_book_pages_l190_190318

-- Conditions
def science_pages : ℕ := 600
def novel_pages (science: ℕ) : ℕ := science / 4
def history_pages (novel: ℕ) : ℕ := novel * 2

-- Theorem to prove
theorem history_book_pages : history_pages (novel_pages science_pages) = 300 :=
by
  sorry

end history_book_pages_l190_190318


namespace circle_center_line_condition_l190_190252

theorem circle_center_line_condition (a : ℝ) :
    (∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 4 * y - 6 = 0 → (a, -2) = (x, y) → x + 2 * y + 1 = 0) → a = 3 :=
by
  sorry

end circle_center_line_condition_l190_190252


namespace max_coins_as_pleases_max_coins_equally_distributed_l190_190636

-- Part a
theorem max_coins_as_pleases {N : ℕ} (N_warriors : N = 33) (total_coins : ℕ := 240) : 
  ∃ k : ℕ, k ≤ N ∧ (∃ remaining_coins : ℕ, remaining_coins ≤ total_coins ∧ remaining_coins = 31) := 
by
  sorry

-- Part b
theorem max_coins_equally_distributed {N : ℕ} (N_warriors : N = 33) (total_coins : ℕ := 240) : 
  ∃ k : ℕ, k ≤ N ∧ (∃ remaining_coins : ℕ, remaining_coins ≤ total_coins ∧ remaining_coins = 30) := 
by
  sorry

end max_coins_as_pleases_max_coins_equally_distributed_l190_190636


namespace six_letter_words_no_substring_amc_l190_190974

theorem six_letter_words_no_substring_amc : 
  let alphabet := ['A', 'M', 'C']
  let totalNumberOfWords := 3^6
  let numberOfWordsContainingAMC := 4 * 3^3 - 1
  let numberOfWordsNotContainingAMC := totalNumberOfWords - numberOfWordsContainingAMC
  numberOfWordsNotContainingAMC = 622 :=
by
  sorry

end six_letter_words_no_substring_amc_l190_190974


namespace find_side_and_area_l190_190137

-- Conditions
variables {A B C a b c : ℝ} (S : ℝ)
axiom angle_sum : A + B + C = Real.pi
axiom side_a : a = 4
axiom side_b : b = 5
axiom angle_relation : C = 2 * A

-- Proven equalities
theorem find_side_and_area :
  c = 6 ∧ S = 5 * 6 * (Real.sqrt 7) / 4 / 2 := by
  sorry

end find_side_and_area_l190_190137


namespace sum_of_a_and_b_l190_190430

noncomputable def log_function (a b x : ℝ) : ℝ := Real.log (x + b) / Real.log a

theorem sum_of_a_and_b (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : log_function a b 2 = 1)
                      (h4 : ∃ x : ℝ, log_function a b x = 8 ∧ log_function a b x = 2) :
  a + b = 4 :=
by
  sorry

end sum_of_a_and_b_l190_190430


namespace coefficient_of_linear_term_l190_190213

def polynomial (x : ℝ) := x^2 - 2 * x - 3

theorem coefficient_of_linear_term : (∀ x : ℝ, polynomial x = x^2 - 2 * x - 3) → -2 = -2 := by
  intro h
  sorry

end coefficient_of_linear_term_l190_190213


namespace gcd_lcm_product_l190_190375

theorem gcd_lcm_product (a b : ℕ) (ha : a = 225) (hb : b = 252) :
  Nat.gcd a b * Nat.lcm a b = 56700 := by
  sorry

end gcd_lcm_product_l190_190375


namespace meaningful_fraction_l190_190230

theorem meaningful_fraction (x : ℝ) : (x + 5 ≠ 0) → (x ≠ -5) :=
by
  sorry

end meaningful_fraction_l190_190230


namespace no_square_ends_in_2012_l190_190968

theorem no_square_ends_in_2012 : ¬ ∃ a : ℤ, (a * a) % 10 = 2 := by
  sorry

end no_square_ends_in_2012_l190_190968


namespace other_team_members_points_l190_190897

theorem other_team_members_points :
  ∃ (x : ℕ), ∃ (y : ℕ), (y ≤ 9 * 3) ∧ (x = y + 18 + x / 3 + x / 5) ∧ y = 24 :=
by
  sorry

end other_team_members_points_l190_190897


namespace knights_rearrangement_impossible_l190_190508

theorem knights_rearrangement_impossible :
  ∀ (b : ℕ → ℕ → Prop), (b 0 0 = true) ∧ (b 0 2 = true) ∧ (b 2 0 = true) ∧ (b 2 2 = true) ∧
  (b 0 0 = b 0 2) ∧ (b 2 0 ≠ b 2 2) → ¬(∃ (b' : ℕ → ℕ → Prop), 
  (b' 0 0 ≠ b 0 0) ∧ (b' 0 2 ≠ b 0 2) ∧ (b' 2 0 ≠ b 2 0) ∧ (b' 2 2 ≠ b 2 2) ∧ 
  (b' 0 0 ≠ b' 0 2) ∧ (b' 2 0 ≠ b' 2 2)) :=
by { sorry }

end knights_rearrangement_impossible_l190_190508


namespace parameter_condition_l190_190395

theorem parameter_condition (a : ℝ) :
  let D := 4 - 4 * a
  let diff_square := ((-2 / a) ^ 2 - 4 * (1 / a))
  D = 9 * diff_square -> a = -3 :=
by
  sorry -- Proof omitted

end parameter_condition_l190_190395


namespace largest_pentagon_angle_is_179_l190_190532

-- Define the interior angles of the pentagon
def angle1 (x : ℝ) := x + 2
def angle2 (x : ℝ) := 2 * x + 3
def angle3 (x : ℝ) := 3 * x - 5
def angle4 (x : ℝ) := 4 * x + 1
def angle5 (x : ℝ) := 5 * x - 1

-- Define the sum of the interior angles of a pentagon
def pentagon_angle_sum := angle1 36 + angle2 36 + angle3 36 + angle4 36 + angle5 36

-- Define the largest angle function
def largest_angle (x : ℝ) := 5 * x - 1

-- The main theorem stating the largest angle measure
theorem largest_pentagon_angle_is_179 (h : angle1 36 + angle2 36 + angle3 36 + angle4 36 + angle5 36 = 540) :
  largest_angle 36 = 179 :=
sorry

end largest_pentagon_angle_is_179_l190_190532


namespace dryer_runtime_per_dryer_l190_190965

-- Definitions for the given conditions
def washer_cost : ℝ := 4
def dryer_cost_per_10min : ℝ := 0.25
def loads_of_laundry : ℕ := 2
def num_dryers : ℕ := 3
def total_spent : ℝ := 11

-- Statement to prove
theorem dryer_runtime_per_dryer : 
  (2 * washer_cost + ((total_spent - 2 * washer_cost) / dryer_cost_per_10min) * 10) / num_dryers = 40 :=
by
  sorry

end dryer_runtime_per_dryer_l190_190965


namespace extended_cross_cannot_form_cube_l190_190390

-- Define what it means to form a cube from patterns
def forms_cube (pattern : Type) : Prop := 
  sorry -- Definition for forming a cube would be detailed here

-- Define the Extended Cross pattern in a way that captures its structure
def extended_cross : Type := sorry -- Definition for Extended Cross structure

-- Define the L shape pattern in a way that captures its structure
def l_shape : Type := sorry -- Definition for L shape structure

-- The theorem statement proving that the Extended Cross pattern cannot form a cube
theorem extended_cross_cannot_form_cube : ¬(forms_cube extended_cross) := 
  sorry

end extended_cross_cannot_form_cube_l190_190390


namespace goldfish_problem_l190_190449

theorem goldfish_problem (x : ℕ) : 
  (18 + (x - 5) * 7 = 4) → (x = 3) :=
by
  intros
  sorry

end goldfish_problem_l190_190449


namespace taxi_speed_is_60_l190_190004

theorem taxi_speed_is_60 (v_b v_t : ℝ) (h1 : v_b = v_t - 30) (h2 : 3 * v_t = 6 * v_b) : v_t = 60 := 
by 
  sorry

end taxi_speed_is_60_l190_190004


namespace keun_bae_jumps_fourth_day_l190_190554

def jumps (n : ℕ) : ℕ :=
  match n with
  | 0 => 15
  | n + 1 => 2 * jumps n

theorem keun_bae_jumps_fourth_day : jumps 3 = 120 :=
by
  sorry

end keun_bae_jumps_fourth_day_l190_190554


namespace phone_calls_to_reach_Davina_l190_190289

theorem phone_calls_to_reach_Davina : 
  (∀ (a b : ℕ), (0 ≤ a ∧ a < 10) ∧ (0 ≤ b ∧ b < 10)) → (least_num_calls : ℕ) = 100 :=
by
  sorry

end phone_calls_to_reach_Davina_l190_190289


namespace sum_of_medians_bounds_l190_190280

theorem sum_of_medians_bounds (a b c m_a m_b m_c : ℝ) 
    (h1 : m_a < (b + c) / 2)
    (h2 : m_b < (a + c) / 2)
    (h3 : m_c < (a + b) / 2)
    (h4 : ∀a b c : ℝ, a + b > c) :
    (3 / 4) * (a + b + c) < m_a + m_b + m_c ∧ m_a + m_b + m_c < a + b + c := 
by
  sorry

end sum_of_medians_bounds_l190_190280


namespace find_a_value_l190_190056

theorem find_a_value (a : ℝ) (f : ℝ → ℝ)
  (h_def : ∀ x, f x = (Real.exp (x - a) - 1) * Real.log (x + 2 * a - 1))
  (h_ge_0 : ∀ x, x > 1 - 2 * a → f x ≥ 0) : a = 2 / 3 :=
by
  -- Omitted proof
  sorry

end find_a_value_l190_190056


namespace cubes_difference_l190_190846

theorem cubes_difference :
  let a := 642
  let b := 641
  a^3 - b^3 = 1234567 :=
by
  let a := 642
  let b := 641
  have h : a^3 - b^3 = 264609288 - 263374721 := sorry
  have h_correct : 264609288 - 263374721 = 1234567 := sorry
  exact Eq.trans h h_correct

end cubes_difference_l190_190846


namespace chord_midpoint_line_l190_190101

open Real 

theorem chord_midpoint_line (x y : ℝ) (P : ℝ × ℝ) 
  (hP : P = (1, 1)) (hcircle : ∀ (x y : ℝ), x^2 + y^2 = 10) :
  x + y - 2 = 0 :=
by
  sorry

end chord_midpoint_line_l190_190101


namespace dot_product_computation_l190_190501

open Real

variables (a b : ℝ) (θ : ℝ)

noncomputable def dot_product (u v : ℝ) : ℝ :=
  u * v * cos θ

noncomputable def magnitude (v : ℝ) : ℝ :=
  abs v

theorem dot_product_computation (a b : ℝ) (h1 : θ = 120) (h2 : magnitude a = 4) (h3 : magnitude b = 4) :
  dot_product b (3 * a + b) = -8 :=
by
  sorry

end dot_product_computation_l190_190501


namespace polygon_with_20_diagonals_is_octagon_l190_190211

theorem polygon_with_20_diagonals_is_octagon :
  ∃ (n : ℕ), n ≥ 3 ∧ (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end polygon_with_20_diagonals_is_octagon_l190_190211


namespace total_washer_dryer_cost_l190_190373

def washer_cost : ℕ := 710
def dryer_cost : ℕ := washer_cost - 220

theorem total_washer_dryer_cost :
  washer_cost + dryer_cost = 1200 :=
  by sorry

end total_washer_dryer_cost_l190_190373


namespace age_of_child_l190_190626

theorem age_of_child (H W C : ℕ) (h1 : (H + W) / 2 = 23) (h2 : (H + 5 + W + 5 + C) / 3 = 19) : C = 1 := by
  sorry

end age_of_child_l190_190626


namespace point_outside_circle_l190_190680

theorem point_outside_circle (a : ℝ) :
  (a > 1) → (a, a) ∉ {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 2 * a * p.1 + a^2 - a = 0} :=
by sorry

end point_outside_circle_l190_190680


namespace length_of_room_calculation_l190_190431

variable (broadness_of_room : ℝ) (width_of_carpet : ℝ) (total_cost : ℝ) (rate_per_sq_meter : ℝ) (area_of_carpet : ℝ) (length_of_room : ℝ)

theorem length_of_room_calculation (h1 : broadness_of_room = 9) 
    (h2 : width_of_carpet = 0.75) 
    (h3 : total_cost = 1872) 
    (h4 : rate_per_sq_meter = 12) 
    (h5 : area_of_carpet = total_cost / rate_per_sq_meter)
    (h6 : area_of_carpet = length_of_room * width_of_carpet) 
    : length_of_room = 208 := 
by 
    sorry

end length_of_room_calculation_l190_190431


namespace remainder_of_k_l190_190984

theorem remainder_of_k {k : ℕ} (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k % 8 = 7) (h4 : k % 11 = 3) (h5 : k < 168) :
  k % 13 = 8 := 
sorry

end remainder_of_k_l190_190984


namespace thirty_percent_more_than_80_is_one_fourth_less_l190_190149

-- Translating the mathematical equivalency conditions into Lean definitions and theorems

def thirty_percent_more (n : ℕ) : ℕ :=
  n + (n * 30 / 100)

def one_fourth_less (x : ℕ) : ℕ :=
  x - (x / 4)

theorem thirty_percent_more_than_80_is_one_fourth_less (x : ℕ) :
  thirty_percent_more 80 = one_fourth_less x → x = 139 :=
by
  sorry

end thirty_percent_more_than_80_is_one_fourth_less_l190_190149


namespace tea_in_each_box_initially_l190_190887

theorem tea_in_each_box_initially (x : ℕ) 
  (h₁ : 4 * (x - 9) = x) : 
  x = 12 := 
sorry

end tea_in_each_box_initially_l190_190887


namespace geometric_sequence_value_l190_190013

variable {a_n : ℕ → ℝ}

-- Condition: {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given: a_1 a_2 a_3 = -8
variable (a1 a2 a3 : ℝ) (h_seq : is_geometric_sequence a_n)
variable (h_cond : a1 * a2 * a3 = -8)

-- Prove: a2 = -2
theorem geometric_sequence_value : a2 = -2 :=
by
  -- Proof will be provided later
  sorry

end geometric_sequence_value_l190_190013


namespace unknown_rate_of_two_towels_l190_190931

theorem unknown_rate_of_two_towels :
  let x := 325
  let known_cost := (3 * 100) + (5 * 150)
  let total_average_price := 170
  let number_of_towels := 10
  known_cost + (2 * x) = total_average_price * number_of_towels :=
by
  let x := 325
  let known_cost := (3 * 100) + (5 * 150)
  let total_average_price := 170
  let number_of_towels := 10
  show known_cost + (2 * x) = total_average_price * number_of_towels
  sorry

end unknown_rate_of_two_towels_l190_190931


namespace geometric_series_r_l190_190397

theorem geometric_series_r (a r : ℝ) 
    (h1 : a * (1 - r ^ 0) / (1 - r) = 24) 
    (h2 : a * r / (1 - r ^ 2) = 8) : 
    r = 1 / 2 := 
sorry

end geometric_series_r_l190_190397


namespace total_cost_of_shoes_before_discount_l190_190704

theorem total_cost_of_shoes_before_discount (S J H : ℝ) (D : ℝ) (shoes jerseys hats : ℝ) :
  jerseys = 1/4 * shoes ∧
  hats = 2 * jerseys ∧
  D = 0.9 * (6 * shoes + 4 * jerseys + 3 * hats) ∧
  D = 620 →
  6 * shoes = 486.30 := by
  sorry

end total_cost_of_shoes_before_discount_l190_190704


namespace minimum_value_of_f_l190_190625

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x

theorem minimum_value_of_f : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = -2 * Real.exp 1 :=
by
  sorry

end minimum_value_of_f_l190_190625


namespace scientific_notation_of_number_l190_190581

def number := 460000000
def scientific_notation (n : ℕ) (s : ℝ) := s * 10 ^ n

theorem scientific_notation_of_number :
  scientific_notation 8 4.6 = number :=
sorry

end scientific_notation_of_number_l190_190581


namespace number_of_sections_l190_190359

def total_seats : ℕ := 270
def seats_per_section : ℕ := 30

theorem number_of_sections : total_seats / seats_per_section = 9 := 
by sorry

end number_of_sections_l190_190359


namespace celery_cost_l190_190682

noncomputable def supermarket_problem
  (total_money : ℕ)
  (price_cereal discount_cereal price_bread : ℕ)
  (price_milk discount_milk price_potato num_potatoes : ℕ)
  (leftover_money : ℕ) 
  (total_cost : ℕ) 
  (cost_of_celery : ℕ) :=
  (price_cereal * discount_cereal / 100 + 
   price_bread + 
   price_milk * discount_milk / 100 + 
   price_potato * num_potatoes) + 
   leftover_money = total_money ∧
  total_cost = total_money - leftover_money ∧
  (price_cereal * discount_cereal / 100 + 
   price_bread + 
   price_milk * discount_milk / 100 + 
   price_potato * num_potatoes) = total_cost - cost_of_celery

theorem celery_cost (total_money : ℕ := 60) 
  (price_cereal : ℕ := 12) 
  (discount_cereal : ℕ := 50) 
  (price_bread : ℕ := 8) 
  (price_milk : ℕ := 10) 
  (discount_milk : ℕ := 90) 
  (price_potato : ℕ := 1) 
  (num_potatoes : ℕ := 6) 
  (leftover_money : ℕ := 26) 
  (total_cost : ℕ := 34) :
  supermarket_problem total_money price_cereal discount_cereal price_bread price_milk discount_milk price_potato num_potatoes leftover_money total_cost 5 :=
by
  sorry

end celery_cost_l190_190682


namespace infection_equation_l190_190838

-- Given conditions
def initially_infected : Nat := 1
def total_after_two_rounds : ℕ := 81
def avg_infect_per_round (x : ℕ) : ℕ := x

-- Mathematically equivalent proof problem
theorem infection_equation (x : ℕ) 
  (h1 : initially_infected = 1)
  (h2 : total_after_two_rounds = 81)
  (h3 : ∀ (y : ℕ), initially_infected + avg_infect_per_round y + (avg_infect_per_round y)^2 = total_after_two_rounds):
  (1 + x)^2 = 81 :=
by
  sorry

end infection_equation_l190_190838


namespace sequence_arithmetic_l190_190906

theorem sequence_arithmetic (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h : ∀ n, S n = 2 * n^2 - 3 * n)
  (h₀ : S 0 = 0) 
  (h₁ : ∀ n, S (n+1) = S n + a (n+1)) :
  ∀ n, a n = 4 * n - 1 := sorry

end sequence_arithmetic_l190_190906


namespace amount_c_gets_l190_190852

theorem amount_c_gets (total_amount : ℕ) (ratio_b ratio_c : ℕ) (h_total_amount : total_amount = 2000) (h_ratio : ratio_b = 4 ∧ ratio_c = 16) : ∃ (c_amount: ℕ), c_amount = 1600 :=
by
  sorry

end amount_c_gets_l190_190852


namespace cristian_cookie_problem_l190_190356

theorem cristian_cookie_problem (white_cookies_init black_cookies_init eaten_black_cookies eaten_white_cookies remaining_black_cookies remaining_white_cookies total_remaining_cookies : ℕ) 
  (h_initial_white : white_cookies_init = 80)
  (h_black_more : black_cookies_init = white_cookies_init + 50)
  (h_eats_half_black : eaten_black_cookies = black_cookies_init / 2)
  (h_eats_three_fourth_white : eaten_white_cookies = (3 / 4) * white_cookies_init)
  (h_remaining_black : remaining_black_cookies = black_cookies_init - eaten_black_cookies)
  (h_remaining_white : remaining_white_cookies = white_cookies_init - eaten_white_cookies)
  (h_total_remaining : total_remaining_cookies = remaining_black_cookies + remaining_white_cookies) :
  total_remaining_cookies = 85 :=
by
  sorry

end cristian_cookie_problem_l190_190356


namespace number_is_square_l190_190074

theorem number_is_square (x y : ℕ) : (∃ n : ℕ, (1100 * x + 11 * y = n^2)) ↔ (x = 7 ∧ y = 4) :=
by
  sorry

end number_is_square_l190_190074


namespace steven_name_day_44_l190_190296

def W (n : ℕ) : ℕ :=
  2 * (n / 2) + 4 * ((n - 1) / 2)

theorem steven_name_day_44 : ∃ n : ℕ, W n = 44 :=
  by 
  existsi 16
  sorry

end steven_name_day_44_l190_190296


namespace jenny_proposal_time_l190_190545

theorem jenny_proposal_time (total_time research_time report_time proposal_time : ℕ) 
  (h1 : total_time = 20) 
  (h2 : research_time = 10) 
  (h3 : report_time = 8) 
  (h4 : proposal_time = total_time - research_time - report_time) : 
  proposal_time = 2 := 
by
  sorry

end jenny_proposal_time_l190_190545


namespace number_of_balls_is_fifty_l190_190565

variable (x : ℝ)
variable (h : x - 40 = 60 - x)

theorem number_of_balls_is_fifty : x = 50 :=
by
  have : 2 * x = 100 := by
    linarith
  linarith

end number_of_balls_is_fifty_l190_190565


namespace total_additions_and_multiplications_l190_190572

def f(x : ℝ) : ℝ := 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 7

theorem total_additions_and_multiplications {x : ℝ} (h : x = 0.6) :
  let horner_f := ((((((6 * x + 5) * x + 4) * x + 3) * x + 2) * x + 1) * x + 7)
  (horner_f = f x) ∧ (6 + 6 = 12) :=
by
  sorry

end total_additions_and_multiplications_l190_190572


namespace greatest_integer_difference_l190_190437

theorem greatest_integer_difference (x y : ℤ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) :
  ∀ d : ℤ, (d = y - x) → d ≤ 6 := 
sorry

end greatest_integer_difference_l190_190437


namespace problem_l190_190949

-- Define what it means to be a factor or divisor
def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a
def is_divisor (a b : ℕ) : Prop := a ∣ b

-- The specific problem conditions
def statement_A := is_factor 4 28
def statement_B := is_divisor 19 209 ∧ ¬ is_divisor 19 57
def statement_C := ¬ is_divisor 30 90 ∧ ¬ is_divisor 30 76
def statement_D := is_divisor 14 28 ∧ ¬ is_divisor 14 56
def statement_E := is_factor 9 162

-- The proof problem
theorem problem : statement_A ∧ ¬statement_B ∧ ¬statement_C ∧ ¬statement_D ∧ statement_E :=
by 
  -- You would normally provide the proof here
  sorry

end problem_l190_190949


namespace binom_8_2_eq_28_l190_190657

open Nat

theorem binom_8_2_eq_28 : Nat.choose 8 2 = 28 := by
  sorry

end binom_8_2_eq_28_l190_190657


namespace problem_statement_l190_190087

noncomputable def universal_set : Set ℤ := {x : ℤ | x^2 - 5*x - 6 < 0 }

def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 2 }

def B : Set ℤ := {2, 3, 5}

def complement_U_A : Set ℤ := {x : ℤ | x ∈ universal_set ∧ ¬(x ∈ A)}

theorem problem_statement : 
  (complement_U_A ∩ B) = {3, 5} :=
by 
  sorry

end problem_statement_l190_190087


namespace find_larger_number_l190_190647

variable (x y : ℕ)

theorem find_larger_number (h1 : 4 * y = 5 * x) (h2 : y - x = 10) : y = 50 := 
by 
  sorry

end find_larger_number_l190_190647


namespace quadratic_real_solution_l190_190789

theorem quadratic_real_solution (m : ℝ) (i : ℂ) (h_i : i * i = -1)
  (h_quad : ∃ z : ℝ, z^2 + (i * z) + m = 0) : m = 0 :=
sorry

end quadratic_real_solution_l190_190789


namespace bridge_length_is_correct_l190_190914

noncomputable def length_of_bridge (length_of_train : ℝ) (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  let total_distance := speed_ms * time_sec
  total_distance - length_of_train

theorem bridge_length_is_correct :
  length_of_bridge 160 45 30 = 215 := by
  sorry

end bridge_length_is_correct_l190_190914


namespace alice_bracelets_given_away_l190_190478

theorem alice_bracelets_given_away
    (total_bracelets : ℕ)
    (cost_of_materials : ℝ)
    (price_per_bracelet : ℝ)
    (profit : ℝ)
    (bracelets_given_away : ℕ)
    (bracelets_sold : ℕ)
    (total_revenue : ℝ)
    (h1 : total_bracelets = 52)
    (h2 : cost_of_materials = 3)
    (h3 : price_per_bracelet = 0.25)
    (h4 : profit = 8)
    (h5 : total_revenue = profit + cost_of_materials)
    (h6 : total_revenue = price_per_bracelet * bracelets_sold)
    (h7 : total_bracelets = bracelets_sold + bracelets_given_away) :
    bracelets_given_away = 8 :=
by
  sorry

end alice_bracelets_given_away_l190_190478


namespace polynomial_solution_l190_190002

noncomputable def f (n : ℕ) (X Y : ℝ) : ℝ :=
  (X - 2 * Y) * (X + Y) ^ (n - 1)

theorem polynomial_solution (n : ℕ) (f : ℝ → ℝ → ℝ)
  (h1 : ∀ (t x y : ℝ), f (t * x) (t * y) = t^n * f x y)
  (h2 : ∀ (a b c : ℝ), f (a + b) c + f (b + c) a + f (c + a) b = 0)
  (h3 : f 1 0 = 1) :
  ∀ (X Y : ℝ), f X Y = (X - 2 * Y) * (X + Y) ^ (n - 1) :=
by
  sorry

end polynomial_solution_l190_190002


namespace percentage_saved_l190_190388

-- Define the actual and saved amount.
def actual_investment : ℕ := 150000
def saved_amount : ℕ := 50000

-- Define the planned investment based on the conditions.
def planned_investment : ℕ := actual_investment + saved_amount

-- Proof goal: The percentage saved is 25%.
theorem percentage_saved : (saved_amount * 100) / planned_investment = 25 := 
by 
  sorry

end percentage_saved_l190_190388


namespace problem_3_equals_answer_l190_190881

variable (a : ℝ)

theorem problem_3_equals_answer :
  (-2 * a^2)^3 / (2 * a^2) = -4 * a^4 :=
by
  sorry

end problem_3_equals_answer_l190_190881


namespace triangle_ratio_l190_190393

theorem triangle_ratio (a b c : ℝ) (P Q : ℝ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c)
  (h₄ : P > 0) (h₅ : Q > P) (h₆ : Q < c) (h₇ : P = 21) (h₈ : Q - P = 35) (h₉ : c - Q = 100)
  (h₁₀ : P + (Q - P) + (c - Q) = c)
  (angle_trisect : ∃ x y : ℝ, x ≠ y ∧ x = a / b ∧ y = 7 / 45) :
  ∃ p q r : ℕ, p + q + r = 92 ∧ p.gcd r = 1 ∧ ¬ ∃ k : ℕ, k^2 ∣ q := sorry

end triangle_ratio_l190_190393


namespace even_and_increasing_on_0_inf_l190_190130

noncomputable def fA (x : ℝ) : ℝ := x^(2/3)
noncomputable def fB (x : ℝ) : ℝ := (1/2)^x
noncomputable def fC (x : ℝ) : ℝ := Real.log x
noncomputable def fD (x : ℝ) : ℝ := -x^2 + 1

theorem even_and_increasing_on_0_inf (f : ℝ → ℝ) : 
  (∀ x, f x = f (-x)) ∧ (∀ a b, (0 < a ∧ a < b) → f a < f b) ↔ f = fA :=
sorry

end even_and_increasing_on_0_inf_l190_190130


namespace log_expression_value_l190_190178

theorem log_expression_value (x : ℝ) (hx : x < 1) (h : (Real.log x / Real.log 10)^3 - 2 * (Real.log (x^3) / Real.log 10) = 150) :
  (Real.log x / Real.log 10)^4 - (Real.log (x^4) / Real.log 10) = 645 := 
sorry

end log_expression_value_l190_190178


namespace log_domain_l190_190250

theorem log_domain (x : ℝ) : 3 - 2 * x > 0 ↔ x < 3 / 2 :=
by
  sorry

end log_domain_l190_190250


namespace solution_set_inequality_l190_190315

theorem solution_set_inequality (x : ℝ) : (1 < x ∧ x < 3) ↔ (x^2 - 4*x + 3 < 0) :=
by sorry

end solution_set_inequality_l190_190315


namespace min_moves_to_checkerboard_l190_190762

noncomputable def minimum_moves_checkerboard (n : ℕ) : ℕ :=
if n = 6 then 18
else 0

theorem min_moves_to_checkerboard :
  minimum_moves_checkerboard 6 = 18 :=
by sorry

end min_moves_to_checkerboard_l190_190762


namespace number_of_bookshelves_l190_190052

theorem number_of_bookshelves (books_in_each: ℕ) (total_books: ℕ) (h_books_in_each: books_in_each = 56) (h_total_books: total_books = 504) : total_books / books_in_each = 9 :=
by
  sorry

end number_of_bookshelves_l190_190052


namespace students_neither_music_nor_art_l190_190462

theorem students_neither_music_nor_art
  (total_students : ℕ) (students_music : ℕ) (students_art : ℕ) (students_both : ℕ)
  (h_total : total_students = 500)
  (h_music : students_music = 30)
  (h_art : students_art = 10)
  (h_both : students_both = 10)
  : total_students - (students_music + students_art - students_both) = 460 :=
by
  rw [h_total, h_music, h_art, h_both]
  norm_num
  sorry

end students_neither_music_nor_art_l190_190462


namespace average_percentage_decrease_l190_190129

theorem average_percentage_decrease (p1 p2 : ℝ) (n : ℕ) (h₀ : p1 = 2000) (h₁ : p2 = 1280) (h₂ : n = 2) :
  ((p1 - p2) / p1 * 100) / n = 18 := 
by
  sorry

end average_percentage_decrease_l190_190129


namespace percent_of_b_l190_190144

variables (a b c : ℝ)

theorem percent_of_b (h1 : c = 0.30 * a) (h2 : b = 1.20 * a) : c = 0.25 * b :=
by sorry

end percent_of_b_l190_190144


namespace simplify_expression_l190_190876

variable (x : ℝ)
variable (h₁ : x ≠ 2)
variable (h₂ : x ≠ 3)
variable (h₃ : x ≠ 4)
variable (h₄ : x ≠ 5)

theorem simplify_expression : 
  ( (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 9) / (x^2 - 8*x + 15)) 
  = ( (x - 1) * (x - 5) ) / ( (x - 4) * (x - 2) * (x - 3) ) ) :=
by sorry

end simplify_expression_l190_190876


namespace lucy_cardinals_vs_blue_jays_l190_190597

noncomputable def day1_cardinals : ℕ := 3
noncomputable def day1_blue_jays : ℕ := 2
noncomputable def day2_cardinals : ℕ := 3
noncomputable def day2_blue_jays : ℕ := 3
noncomputable def day3_cardinals : ℕ := 4
noncomputable def day3_blue_jays : ℕ := 2

theorem lucy_cardinals_vs_blue_jays :
  (day1_cardinals + day2_cardinals + day3_cardinals) - (day1_blue_jays + day2_blue_jays + day3_blue_jays) = 3 :=
  by sorry

end lucy_cardinals_vs_blue_jays_l190_190597


namespace fraction_evaluation_l190_190507

theorem fraction_evaluation : 
  (1/4 - 1/6) / (1/3 - 1/5) = 5/8 := by
  sorry

end fraction_evaluation_l190_190507


namespace hall_volume_proof_l190_190941

-- Define the given conditions.
def hall_length (l : ℝ) : Prop := l = 18
def hall_width (w : ℝ) : Prop := w = 9
def floor_ceiling_area_eq_wall_area (h l w : ℝ) : Prop := 
  2 * (l * w) = 2 * (l * h) + 2 * (w * h)

-- Define the volume calculation.
def hall_volume (l w h V : ℝ) : Prop := 
  V = l * w * h

-- The main theorem stating that the volume is 972 cubic meters.
theorem hall_volume_proof (l w h V : ℝ) 
  (length : hall_length l) 
  (width : hall_width w) 
  (fc_eq_wa : floor_ceiling_area_eq_wall_area h l w) 
  (volume : hall_volume l w h V) : 
  V = 972 :=
  sorry

end hall_volume_proof_l190_190941


namespace arrangements_count_l190_190666

-- Definitions of students and grades
inductive Student : Type
| A | B | C | D | E | F
deriving DecidableEq

inductive Grade : Type
| first | second | third
deriving DecidableEq

-- A function to count valid arrangements
def valid_arrangements (assignments : Student → Grade) : Bool :=
  assignments Student.A = Grade.first ∧
  assignments Student.B ≠ Grade.third ∧
  assignments Student.C ≠ Grade.third ∧
  (assignments Student.A = Grade.first) ∧
  ((assignments Student.B = Grade.second ∧ assignments Student.C = Grade.second ∧ 
    (assignments Student.D ≠ Grade.first ∨ assignments Student.E ≠ Grade.first ∨ assignments Student.F ≠ Grade.first)) ∨
   ((assignments Student.B ≠ Grade.second ∨ assignments Student.C ≠ Grade.second) ∧ 
    (assignments Student.B ≠ Grade.first ∨ assignments Student.C ≠ Grade.first)))

theorem arrangements_count : 
  ∃ (count : ℕ), count = 9 ∧
  count = (Nat.card { assign : Student → Grade // valid_arrangements assign } : ℕ) := sorry

end arrangements_count_l190_190666


namespace isosceles_triangle_base_length_l190_190326

theorem isosceles_triangle_base_length (a b : ℝ) (h1 : a = 3 ∨ b = 3) (h2 : a + a + b = 15 ∨ a + b + b = 15) :
  b = 3 := 
sorry

end isosceles_triangle_base_length_l190_190326


namespace copy_pages_l190_190403

theorem copy_pages
  (total_cents : ℕ)
  (cost_per_page : ℚ)
  (h_total : total_cents = 2000)
  (h_cost : cost_per_page = 2.5) :
  (total_cents / cost_per_page) = 800 :=
by
  -- This is where the proof would go
  sorry

end copy_pages_l190_190403


namespace remaining_amount_to_be_paid_l190_190933

-- Define the conditions
def deposit_percentage : ℚ := 10 / 100
def deposit_amount : ℚ := 80

-- Define the total purchase price based on the conditions
def total_price : ℚ := deposit_amount / deposit_percentage

-- Define the remaining amount to be paid
def remaining_amount : ℚ := total_price - deposit_amount

-- State the theorem
theorem remaining_amount_to_be_paid : remaining_amount = 720 := by
  sorry

end remaining_amount_to_be_paid_l190_190933


namespace problem_a2014_l190_190582

-- Given conditions
def seq (a : ℕ → ℕ) := a 1 = 1 ∧ ∀ n, a (n + 1) = a n + 1

-- Prove the required statement
theorem problem_a2014 (a : ℕ → ℕ) (h : seq a) : a 2014 = 2014 :=
by sorry

end problem_a2014_l190_190582


namespace find_third_number_x_l190_190267

variable {a b : ℝ}

theorem find_third_number_x (h : a < b) :
  (∃ x : ℝ, x = a * b / (2 * b - a) ∧ x < a) ∨ 
  (∃ x : ℝ, x = 2 * a * b / (a + b) ∧ a < x ∧ x < b) ∨ 
  (∃ x : ℝ, x = a * b / (2 * a - b) ∧ a < b ∧ b < x) :=
sorry

end find_third_number_x_l190_190267


namespace price_per_pound_salt_is_50_l190_190758

-- Given conditions
def totalWeight : ℕ := 60
def weightSalt1 : ℕ := 20
def priceSalt2 : ℕ := 35
def weightSalt2 : ℕ := 40
def sellingPricePerPound : ℕ := 48
def desiredProfitRate : ℚ := 0.20

-- Mathematical definitions derived from conditions
def costSalt1 (priceSalt1 : ℕ) : ℕ := weightSalt1 * priceSalt1
def costSalt2 : ℕ := weightSalt2 * priceSalt2
def totalCost (priceSalt1 : ℕ) : ℕ := costSalt1 priceSalt1 + costSalt2
def totalRevenue : ℕ := totalWeight * sellingPricePerPound
def profit (priceSalt1 : ℕ) : ℚ := desiredProfitRate * totalCost priceSalt1
def totalProfit (priceSalt1 : ℕ) : ℚ := totalCost priceSalt1 + profit priceSalt1

-- Proof statement
theorem price_per_pound_salt_is_50 : ∃ (priceSalt1 : ℕ), totalRevenue = totalProfit priceSalt1 ∧ priceSalt1 = 50 := by
  -- We provide the prove structure, exact proof steps are skipped with sorry
  sorry

end price_per_pound_salt_is_50_l190_190758


namespace box_volume_80_possible_l190_190761

theorem box_volume_80_possible :
  ∃ (x : ℕ), 10 * x^3 = 80 :=
by
  sorry

end box_volume_80_possible_l190_190761


namespace parent_payment_per_year_l190_190471

noncomputable def former_salary : ℕ := 45000
noncomputable def raise_percentage : ℕ := 20
noncomputable def number_of_kids : ℕ := 9

theorem parent_payment_per_year : 
  (former_salary + (raise_percentage * former_salary / 100)) / number_of_kids = 6000 := by
  sorry

end parent_payment_per_year_l190_190471


namespace equilateral_triangle_area_perimeter_l190_190219

theorem equilateral_triangle_area_perimeter (altitude : ℝ) : 
  altitude = Real.sqrt 12 →
  (exists area perimeter : ℝ, area = 4 * Real.sqrt 3 ∧ perimeter = 12) :=
by
  intro h_alt
  sorry

end equilateral_triangle_area_perimeter_l190_190219


namespace sides_equal_max_diagonal_at_most_two_l190_190534

variable {n : ℕ}
variable (P : Polygon n)
variable (is_convex : P.IsConvex)
variable (max_diagonal : ℝ)
variable (sides_equal_max_diagonal : list ℝ)
variable (length_sides_equal_max_diagonal : sides_equal_max_diagonal.length)

-- Here we assume the basic conditions given in the problem:
-- 1. The polygon P is convex.
-- 2. The number of sides equal to the longest diagonal are stored in sides_equal_max_diagonal.

theorem sides_equal_max_diagonal_at_most_two :
  is_convex → length_sides_equal_max_diagonal ≤ 2 :=
by
  sorry

end sides_equal_max_diagonal_at_most_two_l190_190534


namespace min_value_of_expression_l190_190097

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h_eq : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3)

theorem min_value_of_expression :
  2 * a + b + c ≥ 2 * Real.sqrt 3 - 2 := by
  sorry

end min_value_of_expression_l190_190097


namespace function_C_is_odd_and_decreasing_l190_190051

-- Conditions
def f (x : ℝ) : ℝ := -x^3 - x

-- Odd function condition
def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

-- Strictly decreasing condition
def is_strictly_decreasing (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2

-- The theorem we want to prove
theorem function_C_is_odd_and_decreasing : 
  is_odd f ∧ is_strictly_decreasing f :=
by
  sorry

end function_C_is_odd_and_decreasing_l190_190051


namespace math_problem_l190_190224

theorem math_problem :
  -50 * 3 - (-2.5) / 0.1 = -125 := by
sorry

end math_problem_l190_190224


namespace cylinder_from_sector_l190_190734

noncomputable def circle_radius : ℝ := 12
noncomputable def sector_angle : ℝ := 300
noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * 2 * Real.pi * r

noncomputable def is_valid_cylinder (base_radius height : ℝ) : Prop :=
  2 * Real.pi * base_radius = arc_length circle_radius sector_angle ∧ height = circle_radius

theorem cylinder_from_sector :
  is_valid_cylinder 10 12 :=
by
  -- here, the proof will be provided
  sorry

end cylinder_from_sector_l190_190734


namespace total_students_sum_is_90_l190_190952

theorem total_students_sum_is_90:
  ∃ (x y z : ℕ), 
  (80 * x - 100 = 92 * (x - 5)) ∧
  (75 * y - 150 = 85 * (y - 6)) ∧
  (70 * z - 120 = 78 * (z - 4)) ∧
  (x + y + z = 90) :=
by
  sorry

end total_students_sum_is_90_l190_190952


namespace alice_probability_at_least_one_multiple_of_4_l190_190599

def probability_multiple_of_4 : ℚ :=
  1 - (45 / 60)^3

theorem alice_probability_at_least_one_multiple_of_4 :
  probability_multiple_of_4 = 37 / 64 :=
by
  sorry

end alice_probability_at_least_one_multiple_of_4_l190_190599


namespace average_visitors_per_day_l190_190645

theorem average_visitors_per_day
  (sunday_visitors : ℕ := 540)
  (other_days_visitors : ℕ := 240)
  (days_in_month : ℕ := 30)
  (first_day_is_sunday : Bool := true)
  (result : ℕ := 290) :
  let num_sundays := 5
  let num_other_days := days_in_month - num_sundays
  let total_visitors := num_sundays * sunday_visitors + num_other_days * other_days_visitors
  let average_visitors := total_visitors / days_in_month
  average_visitors = result :=
by
  sorry

end average_visitors_per_day_l190_190645


namespace nick_paints_wall_in_fraction_l190_190935

theorem nick_paints_wall_in_fraction (nick_paint_time wall_paint_time : ℕ) (h1 : wall_paint_time = 60) (h2 : nick_paint_time = 12) : (nick_paint_time * 1 / wall_paint_time = 1 / 5) :=
by
  sorry

end nick_paints_wall_in_fraction_l190_190935


namespace l_shaped_tile_rectangle_multiple_of_8_l190_190723

theorem l_shaped_tile_rectangle_multiple_of_8 (m n : ℕ) 
  (h : ∃ k : ℕ, 4 * k = m * n) : ∃ s : ℕ, m * n = 8 * s :=
by
  sorry

end l_shaped_tile_rectangle_multiple_of_8_l190_190723


namespace fly_total_distance_l190_190568

-- Definitions and conditions
def cyclist_speed : ℝ := 10 -- speed of each cyclist in miles per hour
def initial_distance : ℝ := 50 -- initial distance between the cyclists in miles
def fly_speed : ℝ := 15 -- speed of the fly in miles per hour

-- Statement to prove
theorem fly_total_distance : 
  (cyclist_speed * 2 * initial_distance / (cyclist_speed + cyclist_speed) / fly_speed * fly_speed) = 37.5 :=
by
  -- sorry is used here to skip the proof
  sorry

end fly_total_distance_l190_190568


namespace total_cars_l190_190276

theorem total_cars (yesterday today : ℕ) (h_yesterday : yesterday = 60) (h_today : today = 2 * yesterday) : yesterday + today = 180 := 
sorry

end total_cars_l190_190276


namespace relationship_roots_geometric_progression_l190_190531

theorem relationship_roots_geometric_progression 
  (x y z p q r : ℝ)
  (h1 : x^2 ≠ y^2 ∧ y^2 ≠ z^2 ∧ x^2 ≠ z^2) -- Distinct non-zero numbers
  (h2 : y^2 = x^2 * r)
  (h3 : z^2 = y^2 * r)
  (h4 : x + y + z = p)
  (h5 : x * y + y * z + z * x = q)
  (h6 : x * y * z = r) : r^2 = 1 := sorry

end relationship_roots_geometric_progression_l190_190531


namespace alison_birth_weekday_l190_190026

-- Definitions for the problem conditions
def days_in_week : ℕ := 7

-- John's birth day
def john_birth_weekday : ℕ := 3  -- Assuming Monday=0, Tuesday=1, ..., Wednesday=3, ...

-- Number of days Alison was born later
def days_later : ℕ := 72

-- Proof that the resultant day is Friday
theorem alison_birth_weekday : (john_birth_weekday + days_later) % days_in_week = 5 :=
by
  sorry

end alison_birth_weekday_l190_190026


namespace area_of_R2_l190_190485

theorem area_of_R2
  (a b : ℝ)
  (h1 : b = 3 * a)
  (h2 : a^2 + b^2 = 225) :
  a * b = 135 / 2 :=
by
  sorry

end area_of_R2_l190_190485


namespace continuous_stripe_probability_l190_190660

-- Define the conditions of the tetrahedron and stripe orientations
def tetrahedron_faces : ℕ := 4
def stripe_orientations_per_face : ℕ := 2
def total_stripe_combinations : ℕ := stripe_orientations_per_face ^ tetrahedron_faces
def favorable_stripe_combinations : ℕ := 2 -- Clockwise and Counterclockwise combinations for a continuous stripe

-- Define the probability calculation
def probability_of_continuous_stripe : ℚ :=
  favorable_stripe_combinations / total_stripe_combinations

-- Theorem statement
theorem continuous_stripe_probability : probability_of_continuous_stripe = 1 / 8 :=
by
  -- The proof is omitted for brevity
  sorry

end continuous_stripe_probability_l190_190660


namespace planted_fraction_l190_190593

theorem planted_fraction (a b : ℕ) (hypotenuse : ℚ) (distance_to_hypotenuse : ℚ) (x : ℚ)
  (h_triangle : a = 5 ∧ b = 12 ∧ hypotenuse = 13)
  (h_distance : distance_to_hypotenuse = 3)
  (h_x : x = 39 / 17)
  (h_square_area : x^2 = 1521 / 289)
  (total_area : ℚ) (planted_area : ℚ)
  (h_total_area : total_area = 30)
  (h_planted_area : planted_area = 7179 / 289) :
  planted_area / total_area = 2393 / 2890 :=
by
  sorry

end planted_fraction_l190_190593


namespace remainder_of_power_modulo_l190_190874

theorem remainder_of_power_modulo : (3^2048) % 11 = 5 := by
  sorry

end remainder_of_power_modulo_l190_190874


namespace swimming_time_per_style_l190_190696

theorem swimming_time_per_style (d v1 v2 v3 v4 t: ℝ) 
    (h1: d = 600) 
    (h2: v1 = 45) 
    (h3: v2 = 35) 
    (h4: v3 = 40) 
    (h5: v4 = 30)
    (h6: t = 15) 
    (h7: d / 4 = 150) 
    : (t / 4 = 3.75) :=
by
  sorry

end swimming_time_per_style_l190_190696


namespace general_eq_line_BC_std_eq_circumscribed_circle_ABC_l190_190442

-- Define the points A, B, and C
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (-4, 1)

-- Prove the general equation of line BC is x + 1 = 0
theorem general_eq_line_BC : ∀ x y : ℝ, (x = -1) → y = 2 ∧ (x = -4) → y = 1 → x + 1 = 0 :=
by
  sorry

-- Prove the standard equation of the circumscribed circle of triangle ABC is (x + 5/2)^2 + (y - 3/2)^2 = 5/2
theorem std_eq_circumscribed_circle_ABC :
  ∀ x y : ℝ,
  (x, y) = (A : ℝ × ℝ) ∨ (x, y) = (B : ℝ × ℝ) ∨ (x, y) = (C : ℝ × ℝ) →
  (x + 5/2)^2 + (y - 3/2)^2 = 5/2 :=
by
  sorry

end general_eq_line_BC_std_eq_circumscribed_circle_ABC_l190_190442


namespace evaluate_expression_l190_190374

theorem evaluate_expression (x y z : ℕ) (hx : x = 5) (hy : y = 10) (hz : z = 3) : z * (y - 2 * x) = 0 := by
  sorry

end evaluate_expression_l190_190374


namespace value_of_a_is_minus_one_l190_190021

-- Define the imaginary unit i
def imaginary_unit_i : Complex := Complex.I

-- Define the complex number condition
def complex_number_condition (a : ℝ) : Prop :=
  let z := (a + imaginary_unit_i) / (1 + imaginary_unit_i)
  (Complex.re z) = 0 ∧ (Complex.im z) ≠ 0

-- Prove that the value of the real number a is -1 given the condition
theorem value_of_a_is_minus_one (a : ℝ) (h : complex_number_condition a) : a = -1 :=
sorry

end value_of_a_is_minus_one_l190_190021


namespace neg_proposition_p_l190_190781

variable {x : ℝ}

def proposition_p : Prop := ∀ x ≥ 0, x^3 - 1 ≥ 0

theorem neg_proposition_p : ¬ proposition_p ↔ ∃ x ≥ 0, x^3 - 1 < 0 :=
by sorry

end neg_proposition_p_l190_190781


namespace simplified_expression_l190_190513

theorem simplified_expression :
  (0.2 * 0.4 - 0.3 / 0.5) + (0.6 * 0.8 + 0.1 / 0.2) - 0.9 * (0.3 - 0.2 * 0.4) = 0.262 :=
by
  sorry

end simplified_expression_l190_190513


namespace sam_gave_plums_l190_190526

variable (initial_plums : ℝ) (total_plums : ℝ) (plums_given : ℝ)

theorem sam_gave_plums (h1 : initial_plums = 7.0) (h2 : total_plums = 10.0) (h3 : total_plums = initial_plums + plums_given) :
  plums_given = 3 := 
by
  sorry

end sam_gave_plums_l190_190526


namespace geometric_sequence_sum_l190_190509

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ) (q a2005 a2006 : ℝ), 
    (∀ n, a (n + 1) = a n * q) ∧
    q > 1 ∧
    a2005 + a2006 = 2 ∧ 
    a2005 * a2006 = 3 / 4 ∧ 
    a (2005) = a2005 ∧ 
    a (2006) = a2006 → 
    a (2007) + a (2008) = 18 := 
by
  sorry

end geometric_sequence_sum_l190_190509


namespace sandy_books_second_shop_l190_190993

theorem sandy_books_second_shop (x : ℕ) (h1 : 65 = 1080 / 16) 
                                (h2 : x * 16 = 840) 
                                (h3 : (1080 + 840) / 16 = 120) : 
                                x = 55 :=
by
  sorry

end sandy_books_second_shop_l190_190993


namespace fraction_of_pianists_got_in_l190_190631

-- Define the conditions
def flutes_got_in (f : ℕ) := f = 16
def clarinets_got_in (c : ℕ) := c = 15
def trumpets_got_in (t : ℕ) := t = 20
def total_band_members (total : ℕ) := total = 53
def total_pianists (p : ℕ) := p = 20

-- The main statement we want to prove
theorem fraction_of_pianists_got_in : 
  ∃ (pi : ℕ), 
    flutes_got_in 16 ∧ 
    clarinets_got_in 15 ∧ 
    trumpets_got_in 20 ∧ 
    total_band_members 53 ∧ 
    total_pianists 20 ∧ 
    pi / 20 = 1 / 10 := 
  sorry

end fraction_of_pianists_got_in_l190_190631


namespace razorback_shop_jersey_revenue_l190_190890

theorem razorback_shop_jersey_revenue :
  let price_per_tshirt := 67
  let price_per_jersey := 165
  let tshirts_sold := 74
  let jerseys_sold := 156
  jerseys_sold * price_per_jersey = 25740 := by
  sorry

end razorback_shop_jersey_revenue_l190_190890


namespace cookie_baking_l190_190558

/-- It takes 7 minutes to bake 1 pan of cookies. In 28 minutes, you can bake 4 pans of cookies. -/
theorem cookie_baking (bake_time_per_pan : ℕ) (total_time : ℕ) (num_pans : ℕ) 
  (h1 : bake_time_per_pan = 7)
  (h2 : total_time = 28) : 
  num_pans = 4 := 
by
  sorry

end cookie_baking_l190_190558


namespace base_rate_second_telephone_company_l190_190208

theorem base_rate_second_telephone_company : 
  ∃ B : ℝ, (11 + 20 * 0.25 = B + 20 * 0.20) ∧ B = 12 := by
  sorry

end base_rate_second_telephone_company_l190_190208


namespace total_oranges_l190_190312

def monday_oranges : ℕ := 100
def tuesday_oranges : ℕ := 3 * monday_oranges
def wednesday_oranges : ℕ := 70

theorem total_oranges : monday_oranges + tuesday_oranges + wednesday_oranges = 470 := by
  sorry

end total_oranges_l190_190312


namespace min_value_frac_expr_l190_190709

theorem min_value_frac_expr (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : a < 1) (h₃ : 0 ≤ b) (h₄ : b < 1) (h₅ : 0 ≤ c) (h₆ : c < 1) :
  (1 / ((2 - a) * (2 - b) * (2 - c)) + 1 / ((2 + a) * (2 + b) * (2 + c))) ≥ 1 / 8 :=
sorry

end min_value_frac_expr_l190_190709


namespace problem1_question_problem1_contrapositive_problem1_negation_problem2_question_problem2_contrapositive_problem2_negation_l190_190140

-- Proof statement for problem 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem problem1_question (x y : ℕ) (h : ¬(is_odd x ∧ is_odd y)) : is_odd (x + y) := sorry

theorem problem1_contrapositive (x y : ℕ) (h : is_odd x ∧ is_odd y) : ¬ is_odd (x + y) := sorry

theorem problem1_negation : ∃ (x y : ℕ), ¬(is_odd x ∧ is_odd y) ∧ ¬ is_odd (x + y) := sorry

-- Proof statement for problem 2

structure Square : Type := (is_rhombus : Prop)

def all_squares_are_rhombuses : Prop := ∀ (sq : Square), sq.is_rhombus

theorem problem2_question : all_squares_are_rhombuses = true := sorry

theorem problem2_contrapositive : ¬ all_squares_are_rhombuses = false := sorry

theorem problem2_negation : ¬(∃ (sq : Square), ¬ sq.is_rhombus) = false := sorry

end problem1_question_problem1_contrapositive_problem1_negation_problem2_question_problem2_contrapositive_problem2_negation_l190_190140


namespace Jake_width_proof_l190_190504

-- Define the dimensions of Sara's birdhouse in feet
def Sara_width_feet := 1
def Sara_height_feet := 2
def Sara_depth_feet := 2

-- Convert the dimensions to inches
def Sara_width_inch := Sara_width_feet * 12
def Sara_height_inch := Sara_height_feet * 12
def Sara_depth_inch := Sara_depth_feet * 12

-- Calculate Sara's birdhouse volume
def Sara_volume := Sara_width_inch * Sara_height_inch * Sara_depth_inch

-- Define the dimensions of Jake's birdhouse in inches
def Jake_height_inch := 20
def Jake_depth_inch := 18
def Jake_volume (Jake_width_inch : ℝ) := Jake_width_inch * Jake_height_inch * Jake_depth_inch

-- Difference in volume
def volume_difference := 1152

-- Prove the width of Jake's birdhouse
theorem Jake_width_proof : ∃ (W : ℝ), Jake_volume W - Sara_volume = volume_difference ∧ W = 22.4 := by
  sorry

end Jake_width_proof_l190_190504


namespace sum_consecutive_powers_of_2_divisible_by_6_l190_190162

theorem sum_consecutive_powers_of_2_divisible_by_6 (n : ℕ) :
  ∃ k : ℕ, 2^n + 2^(n+1) = 6 * k :=
sorry

end sum_consecutive_powers_of_2_divisible_by_6_l190_190162


namespace max_fraction_l190_190885

theorem max_fraction (x y : ℝ) (h1 : -6 ≤ x) (h2 : x ≤ -3) (h3 : 3 ≤ y) (h4 : y ≤ 5) :
  (∀ x y, -6 ≤ x → x ≤ -3 → 3 ≤ y → y ≤ 5 → (x - y) / y ≤ -2) :=
by
  sorry

end max_fraction_l190_190885


namespace base_p_prime_values_zero_l190_190768

theorem base_p_prime_values_zero :
  (∀ p : ℕ, p.Prime → 2008 * p^3 + 407 * p^2 + 214 * p + 226 = 243 * p^2 + 382 * p + 471 → False) :=
by
  sorry

end base_p_prime_values_zero_l190_190768


namespace sticks_needed_for_4x4_square_largest_square_with_100_sticks_l190_190518

-- Problem a)
def sticks_needed_for_square (n: ℕ) : ℕ := 2 * n * (n + 1)

theorem sticks_needed_for_4x4_square : sticks_needed_for_square 4 = 40 :=
by
  sorry

-- Problem b)
def max_square_side_length (total_sticks : ℕ) : ℕ × ℕ :=
  let n := Nat.sqrt (total_sticks / 2)
  if 2*n*(n+1) <= total_sticks then (n, total_sticks - 2*n*(n+1)) else (n-1, total_sticks - 2*(n-1)*n)

theorem largest_square_with_100_sticks : max_square_side_length 100 = (6, 16) :=
by
  sorry

end sticks_needed_for_4x4_square_largest_square_with_100_sticks_l190_190518


namespace original_number_is_40_l190_190550

theorem original_number_is_40 (x : ℝ) (h : 1.25 * x - 0.70 * x = 22) : x = 40 :=
by
  sorry

end original_number_is_40_l190_190550


namespace negation_P_l190_190695

variable (P : Prop) (P_def : ∀ x : ℝ, Real.sin x ≤ 1)

theorem negation_P : ¬P ↔ ∃ x : ℝ, Real.sin x > 1 := by
  sorry

end negation_P_l190_190695


namespace necessary_but_not_sufficient_l190_190649

variables {a b : ℝ}

theorem necessary_but_not_sufficient (h : a > 0) (h₁ : a > b) (h₂ : a⁻¹ > b⁻¹) : 
  b < 0 :=
sorry

end necessary_but_not_sufficient_l190_190649


namespace largest_of_five_consecutive_odd_integers_with_product_93555_l190_190291

theorem largest_of_five_consecutive_odd_integers_with_product_93555 : 
  ∃ n, (n * (n + 2) * (n + 4) * (n + 6) * (n + 8) = 93555) ∧ (n + 8 = 19) :=
sorry

end largest_of_five_consecutive_odd_integers_with_product_93555_l190_190291


namespace calculate_sheep_l190_190677

-- Conditions as definitions
def cows : Nat := 24
def goats : Nat := 113
def total_animals_to_transport (groups size_per_group : Nat) : Nat := groups * size_per_group
def cows_and_goats (cows goats : Nat) : Nat := cows + goats

-- The problem statement: Calculate the number of sheep such that the total number of animals matches the target.
theorem calculate_sheep
  (groups : Nat) (size_per_group : Nat) (cows goats : Nat) (transportation_total animals_present : Nat) 
  (h1 : groups = 3) (h2 : size_per_group = 48) (h3 : cows = 24) (h4 : goats = 113) 
  (h5 : animals_present = cows + goats) (h6 : transportation_total = groups * size_per_group) :
  transportation_total - animals_present = 7 :=
by 
  -- To be proven 
  sorry

end calculate_sheep_l190_190677


namespace paint_brush_ratio_l190_190028

theorem paint_brush_ratio 
  (s w : ℝ) 
  (h1 : s > 0) 
  (h2 : w > 0) 
  (h3 : (1 / 2) * w ^ 2 + (1 / 2) * (s - w) ^ 2 = (s ^ 2) / 3) 
  : s / w = 3 + Real.sqrt 3 :=
sorry

end paint_brush_ratio_l190_190028


namespace color_guard_team_row_length_l190_190367

theorem color_guard_team_row_length (n : ℕ) (p d : ℝ)
  (h_n : n = 40)
  (h_p : p = 0.4)
  (h_d : d = 0.5) :
  (n - 1) * d + n * p = 35.5 :=
by
  sorry

end color_guard_team_row_length_l190_190367


namespace min_vitamins_sold_l190_190174

theorem min_vitamins_sold (n : ℕ) (h1 : n % 11 = 0) (h2 : n % 23 = 0) (h3 : n % 37 = 0) : n = 9361 :=
by
  sorry

end min_vitamins_sold_l190_190174


namespace trace_bag_weight_l190_190616

-- Define the weights of Gordon's bags
def gordon_bag1_weight : ℕ := 3
def gordon_bag2_weight : ℕ := 7

-- Define the number of Trace's bags
def trace_num_bags : ℕ := 5

-- Define what we are trying to prove: the weight of one of Trace's shopping bags
theorem trace_bag_weight :
  (gordon_bag1_weight + gordon_bag2_weight) = (trace_num_bags * 2) :=
by
  sorry

end trace_bag_weight_l190_190616


namespace p_sufficient_not_necessary_for_q_l190_190823

-- Define the propositions p and q based on the given conditions
def p (α : ℝ) : Prop := α = Real.pi / 4
def q (α : ℝ) : Prop := Real.sin α = Real.cos α

-- Theorem that states p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q (α : ℝ) : p α → (q α) ∧ ¬(q α → p α) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l190_190823


namespace smallest_number_of_cookies_proof_l190_190700

def satisfies_conditions (a : ℕ) : Prop :=
  (a % 6 = 5) ∧ (a % 8 = 6) ∧ (a % 10 = 9) ∧ (∃ n : ℕ, a = n * n)

def smallest_number_of_cookies : ℕ :=
  2549

theorem smallest_number_of_cookies_proof :
  satisfies_conditions smallest_number_of_cookies :=
by
  sorry

end smallest_number_of_cookies_proof_l190_190700


namespace mode_of_data_set_is_60_l190_190383

theorem mode_of_data_set_is_60
  (data : List ℕ := [65, 60, 75, 60, 80])
  (mode : ℕ := 60) :
  mode = 60 ∧ (∀ x ∈ data, data.count x ≤ data.count 60) :=
by {
  sorry
}

end mode_of_data_set_is_60_l190_190383


namespace average_monthly_growth_rate_proof_profit_in_may_proof_l190_190917

theorem average_monthly_growth_rate_proof :
  ∃ r : ℝ, 2400 * (1 + r)^2 = 3456 ∧ r = 0.2 := sorry

theorem profit_in_may_proof (r : ℝ) (h_r : r = 0.2) :
  3456 * (1 + r) = 4147.2 := sorry

end average_monthly_growth_rate_proof_profit_in_may_proof_l190_190917


namespace no_positive_real_solution_l190_190826

open Real

theorem no_positive_real_solution (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  ¬(∀ n : ℕ, 0 < n → (n - 2) / a ≤ ⌊b * n⌋ ∧ ⌊b * n⌋ < (n - 1) / a) :=
by sorry

end no_positive_real_solution_l190_190826


namespace max_value_of_g_is_34_l190_190590
noncomputable def g : ℕ → ℕ
| n => if n < 15 then n + 20 else g (n - 7)

theorem max_value_of_g_is_34 : ∃ n, g n = 34 ∧ ∀ m, g m ≤ 34 :=
by
  sorry

end max_value_of_g_is_34_l190_190590


namespace problem_real_numbers_inequality_l190_190892

open Real

theorem problem_real_numbers_inequality 
  (a1 b1 a2 b2 : ℝ) :
  a1 * b1 + a2 * b2 ≤ sqrt (a1^2 + a2^2) * sqrt (b1^2 + b2^2) :=
by 
  sorry

end problem_real_numbers_inequality_l190_190892


namespace fraction_expression_evaluation_l190_190617

theorem fraction_expression_evaluation : 
  (1/4 - 1/6) / (1/3 - 1/4) = 1 := 
by
  sorry

end fraction_expression_evaluation_l190_190617


namespace sequence_sum_l190_190116

theorem sequence_sum (A B C D E F G H : ℕ) (hC : C = 7) 
    (h_sum : A + B + C = 36 ∧ B + C + D = 36 ∧ C + D + E = 36 ∧ D + E + F = 36 ∧ E + F + G = 36 ∧ F + G + H = 36) :
    A + H = 29 :=
sorry

end sequence_sum_l190_190116


namespace steel_ingot_weight_l190_190257

theorem steel_ingot_weight 
  (initial_weight : ℕ)
  (percent_increase : ℚ)
  (ingot_cost : ℚ)
  (discount_threshold : ℕ)
  (discount_percent : ℚ)
  (total_cost : ℚ)
  (added_weight : ℚ)
  (number_of_ingots : ℕ)
  (ingot_weight : ℚ)
  (h1 : initial_weight = 60)
  (h2 : percent_increase = 0.6)
  (h3 : ingot_cost = 5)
  (h4 : discount_threshold = 10)
  (h5 : discount_percent = 0.2)
  (h6 : total_cost = 72)
  (h7 : added_weight = initial_weight * percent_increase)
  (h8 : added_weight = ingot_weight * number_of_ingots)
  (h9 : total_cost = (ingot_cost * number_of_ingots) * (1 - discount_percent)) :
  ingot_weight = 2 := 
by
  sorry

end steel_ingot_weight_l190_190257


namespace custom_operation_example_l190_190102

def custom_operation (x y : Int) : Int :=
  x * y - 3 * x

theorem custom_operation_example : (custom_operation 7 4) - (custom_operation 4 7) = -9 := by
  sorry

end custom_operation_example_l190_190102


namespace perimeter_smallest_square_l190_190298

theorem perimeter_smallest_square 
  (d : ℝ) (side_largest : ℝ)
  (h1 : d = 3) 
  (h2 : side_largest = 22) : 
  4 * (side_largest - 2 * d - 2 * d) = 40 := by
  sorry

end perimeter_smallest_square_l190_190298


namespace sum_of_a_b_vert_asymptotes_l190_190073

theorem sum_of_a_b_vert_asymptotes (a b : ℝ) 
  (h1 : ∀ x : ℝ, x = -1 → x^2 + a * x + b = 0) 
  (h2 : ∀ x : ℝ, x = 3 → x^2 + a * x + b = 0) : 
  a + b = -5 :=
sorry

end sum_of_a_b_vert_asymptotes_l190_190073


namespace arithmetic_sequence_problem_l190_190039

theorem arithmetic_sequence_problem (a₁ d S₁₀ : ℝ) (h1 : d < 0) (h2 : (a₁ + d) * (a₁ + 3 * d) = 12) 
  (h3 : (a₁ + d) + (a₁ + 3 * d) = 8) (h4 : S₁₀ = 10 * a₁ + 10 * (10 - 1) / 2 * d) : 
  true := sorry

end arithmetic_sequence_problem_l190_190039


namespace wilted_flowers_correct_l190_190574

-- Definitions based on the given conditions
def total_flowers := 45
def flowers_per_bouquet := 5
def bouquets_made := 2

-- Calculating the number of flowers used for bouquets
def used_flowers : ℕ := bouquets_made * flowers_per_bouquet

-- Question: How many flowers wilted before the wedding?
-- Statement: Prove the number of wilted flowers is 35.
theorem wilted_flowers_correct : total_flowers - used_flowers = 35 := by
  sorry

end wilted_flowers_correct_l190_190574


namespace problem_solution_l190_190179

-- Definitions of odd function and given conditions.
variables {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x) (h_eq : f 3 - f 2 = 1)

-- Proof statement of the math problem.
theorem problem_solution : f (-2) - f (-3) = 1 :=
by
  sorry

end problem_solution_l190_190179


namespace solve_for_a_l190_190088

theorem solve_for_a (a x : ℝ) (h : (1 / 2) * x + a = -1) (hx : x = 2) : a = -2 :=
by
  sorry

end solve_for_a_l190_190088


namespace right_triangle_longer_leg_l190_190981

theorem right_triangle_longer_leg (a b c : ℕ) (h₀ : a^2 + b^2 = c^2) (h₁ : c = 65) (h₂ : a < b) : b = 60 :=
sorry

end right_triangle_longer_leg_l190_190981


namespace degree_f_x2_mul_g_x4_l190_190220

open Polynomial

theorem degree_f_x2_mul_g_x4 {f g : Polynomial ℝ} (hf : degree f = 4) (hg : degree g = 5) :
  degree (f.comp (X ^ 2) * g.comp (X ^ 4)) = 28 :=
sorry

end degree_f_x2_mul_g_x4_l190_190220


namespace rope_length_in_cm_l190_190259

-- Define the given conditions
def num_equal_pieces : ℕ := 150
def length_equal_piece_mm : ℕ := 75
def num_remaining_pieces : ℕ := 4
def length_remaining_piece_mm : ℕ := 100

-- Prove that the total length of the rope in centimeters is 1165
theorem rope_length_in_cm : (num_equal_pieces * length_equal_piece_mm + num_remaining_pieces * length_remaining_piece_mm) / 10 = 1165 :=
by
  sorry

end rope_length_in_cm_l190_190259


namespace sum_of_a_b_c_l190_190502

theorem sum_of_a_b_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc1 : a * b + c = 47) (habc2 : b * c + a = 47) (habc3 : a * c + b = 47) : a + b + c = 48 := 
sorry

end sum_of_a_b_c_l190_190502


namespace prove_sum_eq_9_l190_190284

theorem prove_sum_eq_9 (a b : ℝ) (h : i * (a - i) = b - (2 * i) ^ 3) : a + b = 9 :=
by
  sorry

end prove_sum_eq_9_l190_190284


namespace smallest_pos_d_l190_190086

theorem smallest_pos_d (d : ℕ) (h : d > 0) (hd : ∃ k : ℕ, 3150 * d = k * k) : d = 14 := 
by 
  sorry

end smallest_pos_d_l190_190086


namespace min_value_of_sum_of_powers_l190_190641

theorem min_value_of_sum_of_powers (x y : ℝ) (h : x + 3 * y = 1) : 
  2^x + 8^y ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_of_sum_of_powers_l190_190641


namespace current_dogwood_trees_l190_190407

def number_of_trees (X : ℕ) : Prop :=
  X + 61 = 100

theorem current_dogwood_trees (X : ℕ) (h : number_of_trees X) : X = 39 :=
by 
  sorry

end current_dogwood_trees_l190_190407


namespace calculate_result_l190_190753

theorem calculate_result :
  1 - 2 * (Real.sin (Real.pi / 8))^2 = Real.cos (Real.pi / 4) :=
by
  sorry

end calculate_result_l190_190753


namespace diamond_property_C_l190_190724

-- Define the binary operation diamond
def diamond (a b : ℕ) : ℕ := a ^ (2 * b)

theorem diamond_property_C (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) : 
  (diamond a b) ^ n = diamond a (b * n) :=
by
  sorry

end diamond_property_C_l190_190724


namespace cos_neg_3pi_plus_alpha_l190_190394

/-- Given conditions: 
  1. 𝚌𝚘𝚜(3π/2 + α) = -3/5,
  2. α is an angle in the fourth quadrant,
Prove: cos(-3π + α) = -4/5 -/
theorem cos_neg_3pi_plus_alpha (α : Real) (h1 : Real.cos (3 * Real.pi / 2 + α) = -3 / 5) (h2 : 0 ≤ α ∧ α < 2 * Real.pi ∧ Real.sin α < 0) :
  Real.cos (-3 * Real.pi + α) = -4 / 5 := 
sorry

end cos_neg_3pi_plus_alpha_l190_190394


namespace that_remaining_money_l190_190765

section
/-- Initial money in Olivia's wallet --/
def initial_money : ℕ := 53

/-- Money collected from ATM --/
def collected_money : ℕ := 91

/-- Money spent at the supermarket --/
def spent_money : ℕ := collected_money + 39

/-- Remaining money after visiting the supermarket --
Theorem that proves Olivia's remaining money is 14 dollars.
-/
theorem remaining_money : initial_money + collected_money - spent_money = 14 := 
by
  unfold initial_money collected_money spent_money
  simp
  sorry
end

end that_remaining_money_l190_190765


namespace probability_between_lines_l190_190635

def line_l (x : ℝ) : ℝ := -2 * x + 8
def line_m (x : ℝ) : ℝ := -3 * x + 9

theorem probability_between_lines 
  (h1 : ∀ x > 0, line_l x ≥ 0) 
  (h2 : ∀ x > 0, line_m x ≥ 0) 
  (h3 : ∀ x > 0, line_l x < line_m x ∨ line_m x ≤ 0) : 
  (1 / 16 : ℝ) * 100 = 0.16 :=
by
  sorry

end probability_between_lines_l190_190635


namespace intersection_point_ordinate_interval_l190_190688

theorem intersection_point_ordinate_interval:
  ∃ m : ℤ, ∀ x : ℝ, e ^ x = 5 - x → 3 < x ∧ x < 4 :=
by sorry

end intersection_point_ordinate_interval_l190_190688


namespace mike_sold_song_book_for_correct_amount_l190_190884

-- Define the constants for the cost of the trumpet and the net amount spent
def cost_of_trumpet : ℝ := 145.16
def net_amount_spent : ℝ := 139.32

-- Define the amount received from selling the song book
def amount_received_from_selling_song_book : ℝ :=
  cost_of_trumpet - net_amount_spent

-- The theorem stating the amount Mike sold the song book for
theorem mike_sold_song_book_for_correct_amount :
  amount_received_from_selling_song_book = 5.84 :=
sorry

end mike_sold_song_book_for_correct_amount_l190_190884


namespace helga_extra_hours_last_friday_l190_190401

theorem helga_extra_hours_last_friday
  (weekly_articles : ℕ)
  (extra_hours_thursday : ℕ)
  (extra_articles_thursday : ℕ)
  (extra_articles_friday : ℕ)
  (articles_per_half_hour : ℕ)
  (half_hours_per_hour : ℕ)
  (usual_articles_per_day : ℕ)
  (days_per_week : ℕ)
  (articles_last_thursday_plus_friday : ℕ)
  (total_articles : ℕ) :
  (weekly_articles = (usual_articles_per_day * days_per_week)) →
  (extra_hours_thursday = 2) →
  (articles_per_half_hour = 5) →
  (half_hours_per_hour = 2) →
  (usual_articles_per_day = (articles_per_half_hour * 8)) →
  (extra_articles_thursday = (articles_per_half_hour * (extra_hours_thursday * half_hours_per_hour))) →
  (articles_last_thursday_plus_friday = weekly_articles + extra_articles_thursday) →
  (total_articles = 250) →
  (extra_articles_friday = total_articles - articles_last_thursday_plus_friday) →
  (extra_articles_friday = 30) →
  ((extra_articles_friday / articles_per_half_hour) = 6) →
  (3 = (6 / half_hours_per_hour)) :=
by
  intro hw1 hw2 hw3 hw4 hw5 hw6 hw7 hw8 hw9 hw10
  sorry

end helga_extra_hours_last_friday_l190_190401


namespace josh_remaining_marbles_l190_190157

theorem josh_remaining_marbles : 
  let initial_marbles := 19 
  let lost_marbles := 11
  initial_marbles - lost_marbles = 8 := by
  sorry

end josh_remaining_marbles_l190_190157


namespace find_f_neg_2_l190_190737

def f (x : ℝ) : ℝ := sorry -- The actual function f is undefined here.

theorem find_f_neg_2 (h : ∀ x ≠ 0, f (1 / x) + (1 / x) * f (-x) = 2 * x) :
  f (-2) = 7 / 2 :=
sorry

end find_f_neg_2_l190_190737


namespace smallest_n_divisible_l190_190527

theorem smallest_n_divisible (n : ℕ) : 
  (450 ∣ n^3) ∧ (2560 ∣ n^4) ↔ n = 60 :=
by {
  sorry
}

end smallest_n_divisible_l190_190527


namespace determine_m_l190_190448

theorem determine_m (m : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 ↔ m * (x - 1) > x^2 - x) → m = 2 :=
sorry

end determine_m_l190_190448


namespace sum_series_l190_190744

theorem sum_series : ∑' n, (2 * n + 1) / (n * (n + 1) * (n + 2)) = 1 := 
sorry

end sum_series_l190_190744


namespace total_marbles_l190_190043

theorem total_marbles (mary_marbles : ℕ) (joan_marbles : ℕ) (h1 : mary_marbles = 9) (h2 : joan_marbles = 3) : mary_marbles + joan_marbles = 12 := by
  sorry

end total_marbles_l190_190043


namespace min_value_frac_ineq_l190_190587

theorem min_value_frac_ineq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) :
  ∃ x, x = (1/a) + (2/b) ∧ x ≥ 9 :=
sorry

end min_value_frac_ineq_l190_190587


namespace medical_team_combinations_l190_190701

-- Number of combinations function
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem medical_team_combinations :
  let maleDoctors := 6
  let femaleDoctors := 5
  let requiredMale := 2
  let requiredFemale := 1
  choose maleDoctors requiredMale * choose femaleDoctors requiredFemale = 75 :=
by
  sorry

end medical_team_combinations_l190_190701


namespace range_of_a_l190_190994

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 2| + 1) ↔ 1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l190_190994


namespace ratio_of_erasers_l190_190092

theorem ratio_of_erasers (a n : ℕ) (ha : a = 4) (hn : n = a + 12) :
  n / a = 4 :=
by
  sorry

end ratio_of_erasers_l190_190092


namespace cost_of_each_gumdrop_l190_190945

theorem cost_of_each_gumdrop (cents : ℕ) (gumdrops : ℕ) (cost_per_gumdrop : ℕ) : 
  cents = 224 → gumdrops = 28 → cost_per_gumdrop = cents / gumdrops → cost_per_gumdrop = 8 :=
by
  intros h_cents h_gumdrops h_cost
  sorry

end cost_of_each_gumdrop_l190_190945


namespace smallest_value_geq_4_l190_190623

noncomputable def smallest_value (a b c d : ℝ) : ℝ :=
  (a + b + c + d) * ((1 / (a + b + d)) + (1 / (a + c + d)) + (1 / (b + c + d)))

theorem smallest_value_geq_4 (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  smallest_value a b c d ≥ 4 :=
by
  sorry

end smallest_value_geq_4_l190_190623


namespace leo_third_part_time_l190_190190

theorem leo_third_part_time :
  ∃ (T3 : ℕ), 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 3 → T = 25 * k) →
  T1 = 25 →
  T2 = 50 →
  Break1 = 10 →
  Break2 = 15 →
  TotalTime = 2 * 60 + 30 →
  (TotalTime - (T1 + Break1 + T2 + Break2) = T3) →
  T3 = 50 := 
sorry

end leo_third_part_time_l190_190190


namespace max_volume_is_correct_l190_190825

noncomputable def max_volume_of_inscribed_sphere (AB BC AA₁ : ℝ) (h₁ : AB = 6) (h₂ : BC = 8) (h₃ : AA₁ = 3) : ℝ :=
  let AC := Real.sqrt ((6 : ℝ) ^ 2 + (8 : ℝ) ^ 2)
  let r := (AB + BC - AC) / 2
  let sphere_radius := AA₁ / 2
  (4/3) * Real.pi * sphere_radius ^ 3

theorem max_volume_is_correct : max_volume_of_inscribed_sphere 6 8 3 (by rfl) (by rfl) (by rfl) = 9 * Real.pi / 2 := by
  sorry

end max_volume_is_correct_l190_190825


namespace g_constant_term_l190_190859

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := f * g

-- Conditions from the problem
def f_has_constant_term_5 : f.coeff 0 = 5 := sorry
def h_has_constant_term_neg_10 : h.coeff 0 = -10 := sorry
def g_is_quadratic : g.degree ≤ 2 := sorry

-- Statement of the problem
theorem g_constant_term : g.coeff 0 = -2 :=
by
  have h_eq_fg : h = f * g := rfl
  have f_const := f_has_constant_term_5
  have h_const := h_has_constant_term_neg_10
  have g_quad := g_is_quadratic
  sorry

end g_constant_term_l190_190859


namespace unique_representation_l190_190864

theorem unique_representation {p x y : ℕ} 
  (hp : p > 2 ∧ Prime p) 
  (h : 2 * y = p * (x + y)) 
  (hx : x ≠ y) : 
  ∃ x y : ℕ, (1/x + 1/y = 2/p) ∧ x ≠ y := 
sorry

end unique_representation_l190_190864


namespace tangent_line_inclination_range_l190_190755

theorem tangent_line_inclination_range:
  ∀ (x : ℝ), -1/2 ≤ x ∧ x ≤ 1/2 → (0 ≤ 2*x ∧ 2*x ≤ 1 ∨ -1 ≤ 2*x ∧ 2*x < 0) →
    ∃ (α : ℝ), (0 ≤ α ∧ α ≤ π/4) ∨ (3*π/4 ≤ α ∧ α < π) :=
sorry

end tangent_line_inclination_range_l190_190755


namespace tan_sum_formula_l190_190613

theorem tan_sum_formula {A B : ℝ} (hA : A = 55) (hB : B = 65) (h1 : Real.tan (A + B) = Real.tan 120) 
    (h2 : Real.tan 120 = -Real.sqrt 3) :
    Real.tan 55 + Real.tan 65 - Real.sqrt 3 * Real.tan 55 * Real.tan 65 = -Real.sqrt 3 := 
by
  sorry

end tan_sum_formula_l190_190613


namespace area_of_larger_square_l190_190622

theorem area_of_larger_square (side_length : ℕ) (num_squares : ℕ)
  (h₁ : side_length = 2)
  (h₂ : num_squares = 8) : 
  (num_squares * side_length^2) = 32 :=
by
  sorry

end area_of_larger_square_l190_190622


namespace lizz_team_loses_by_8_points_l190_190321

-- Definitions of the given conditions
def initial_deficit : ℕ := 20
def free_throw_points : ℕ := 5 * 1
def three_pointer_points : ℕ := 3 * 3
def jump_shot_points : ℕ := 4 * 2
def liz_points : ℕ := free_throw_points + three_pointer_points + jump_shot_points
def other_team_points : ℕ := 10
def points_caught_up : ℕ := liz_points - other_team_points
def final_deficit : ℕ := initial_deficit - points_caught_up

-- Theorem proving Liz's team loses by 8 points
theorem lizz_team_loses_by_8_points : final_deficit = 8 :=
  by
    -- Proof will be here
    sorry

end lizz_team_loses_by_8_points_l190_190321


namespace jump_rope_cost_l190_190521

def cost_board_game : ℕ := 12
def cost_playground_ball : ℕ := 4
def saved_money : ℕ := 6
def uncle_money : ℕ := 13
def additional_needed : ℕ := 4

theorem jump_rope_cost :
  let total_money := saved_money + uncle_money
  let total_needed := total_money + additional_needed
  let combined_cost := cost_board_game + cost_playground_ball
  let cost_jump_rope := total_needed - combined_cost
  cost_jump_rope = 7 := by
  sorry

end jump_rope_cost_l190_190521


namespace find_angle_measure_l190_190069

theorem find_angle_measure (x : ℝ) (hx : 90 - x + 40 = (1 / 2) * (180 - x)) : x = 80 :=
by
  sorry

end find_angle_measure_l190_190069


namespace probability_of_target_destroyed_l190_190879

theorem probability_of_target_destroyed :
  let p1 := 0.9
  let p2 := 0.9
  let p3 := 0.8
  (p1 * p2 * p3) + (p1 * p2 * (1 - p3)) + (p1 * (1 - p2) * p3) + ((1 - p1) * p2 * p3) = 0.954 :=
by
  let p1 := 0.9
  let p2 := 0.9
  let p3 := 0.8
  sorry

end probability_of_target_destroyed_l190_190879


namespace arccos_sqrt3_div_2_eq_pi_div_6_l190_190690

theorem arccos_sqrt3_div_2_eq_pi_div_6 : 
  Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 :=
by 
  sorry

end arccos_sqrt3_div_2_eq_pi_div_6_l190_190690


namespace perpendicular_bisectors_intersect_at_one_point_l190_190370

-- Define the key geometric concepts
variables {Point : Type*} [MetricSpace Point]

-- Define the given conditions 
variables (A B C M : Point)
variables (h1 : dist M A = dist M B)
variables (h2 : dist M B = dist M C)

-- Define the theorem to be proven
theorem perpendicular_bisectors_intersect_at_one_point :
  dist M A = dist M C :=
by 
  -- Proof to be filled in later
  sorry

end perpendicular_bisectors_intersect_at_one_point_l190_190370


namespace min_sum_a_b_l190_190783

-- The conditions
variables {a b : ℝ}
variables (h₁ : a > 1) (h₂ : b > 1) (h₃ : ab - (a + b) = 1)

-- The theorem statement
theorem min_sum_a_b : a + b = 2 + 2 * Real.sqrt 2 :=
sorry

end min_sum_a_b_l190_190783


namespace root_quad_eq_sum_l190_190344

theorem root_quad_eq_sum (a b : ℝ) (h1 : a^2 + a - 2022 = 0) (h2 : b^2 + b - 2022 = 0) (h3 : a + b = -1) : a^2 + 2 * a + b = 2021 :=
by sorry

end root_quad_eq_sum_l190_190344


namespace find_fourth_number_l190_190037

theorem find_fourth_number : 
  ∀ (x y : ℝ),
  (28 + x + 42 + y + 104) / 5 = 90 ∧ (128 + 255 + 511 + 1023 + x) / 5 = 423 →
  y = 78 :=
by
  intros x y h
  sorry

end find_fourth_number_l190_190037


namespace sid_money_left_after_purchases_l190_190669

theorem sid_money_left_after_purchases : 
  ∀ (original_money money_spent_on_computer money_spent_on_snacks half_of_original_money money_left final_more_than_half),
  original_money = 48 → 
  money_spent_on_computer = 12 → 
  money_spent_on_snacks = 8 →
  half_of_original_money = original_money / 2 → 
  money_left = original_money - (money_spent_on_computer + money_spent_on_snacks) → 
  final_more_than_half = money_left - half_of_original_money →
  final_more_than_half = 4 := 
by
  intros original_money money_spent_on_computer money_spent_on_snacks half_of_original_money money_left final_more_than_half
  intros h1 h2 h3 h4 h5 h6
  sorry

end sid_money_left_after_purchases_l190_190669


namespace age_problem_l190_190816

theorem age_problem (age x : ℕ) (h : age = 64) :
  (1 / 2 : ℝ) * (8 * (age + x) - 8 * (age - 8)) = age → x = 8 :=
by
  sorry

end age_problem_l190_190816


namespace product_of_base_9_digits_of_9876_l190_190366

def base9_digits (n : ℕ) : List ℕ := 
  let rec digits_aux (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else digits_aux (n / 9) ((n % 9) :: acc)
  digits_aux n []

def product (lst : List ℕ) : ℕ := lst.foldl (· * ·) 1

theorem product_of_base_9_digits_of_9876 :
  product (base9_digits 9876) = 192 :=
by 
  sorry

end product_of_base_9_digits_of_9876_l190_190366


namespace find_k_when_lines_perpendicular_l190_190727

theorem find_k_when_lines_perpendicular (k : ℝ) :
  (∀ x y : ℝ, (k-3) * x + (3-k) * y + 1 = 0 → ∀ x y : ℝ, 2 * (k-3) * x - 2 * y + 3 = 0 → -((k-3)/(3-k)) * (k-3) = -1) → 
  k = 2 :=
by
  sorry

end find_k_when_lines_perpendicular_l190_190727


namespace log_relation_l190_190620

theorem log_relation (a b c: ℝ) (h₁: a = (Real.log 2) / 2) (h₂: b = (Real.log 3) / 3) (h₃: c = (Real.log 5) / 5) : c < a ∧ a < b :=
by
  sorry

end log_relation_l190_190620


namespace bill_experience_l190_190571

theorem bill_experience (B J : ℕ) (h1 : J - 5 = 3 * (B - 5)) (h2 : J = 2 * B) : B = 10 :=
by
  sorry

end bill_experience_l190_190571


namespace cost_of_case_of_rolls_l190_190606

noncomputable def cost_of_multiple_rolls (n : ℕ) (individual_cost : ℝ) : ℝ :=
  n * individual_cost

theorem cost_of_case_of_rolls :
  ∀ (n : ℕ) (C : ℝ) (individual_cost savings_perc : ℝ),
    n = 12 →
    individual_cost = 1 →
    savings_perc = 0.25 →
    C = cost_of_multiple_rolls n (individual_cost * (1 - savings_perc)) →
    C = 9 :=
by
  intros n C individual_cost savings_perc h1 h2 h3 h4
  sorry

end cost_of_case_of_rolls_l190_190606


namespace ellipse_foci_on_y_axis_l190_190251

theorem ellipse_foci_on_y_axis (m : ℝ) : 
  (∃ (x y : ℝ), (x^2 / (m + 2)) - (y^2 / (m + 1)) = 1) ↔ (-2 < m ∧ m < -3/2) := 
by
  sorry

end ellipse_foci_on_y_axis_l190_190251


namespace greatest_multiple_of_5_and_6_less_than_1000_l190_190405

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n ∧ n = 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l190_190405


namespace solution_pairs_count_l190_190022

theorem solution_pairs_count : 
  ∃ (s : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ s → 5 * p.1 + 7 * p.2 = 708) ∧ s.card = 20 :=
sorry

end solution_pairs_count_l190_190022


namespace inequality_proof_l190_190107

noncomputable def a : ℝ := (1 / 2) * Real.cos (6 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (6 * Real.pi / 180)
noncomputable def b : ℝ := (2 * Real.tan (13 * Real.pi / 180)) / (1 - (Real.tan (13 * Real.pi / 180))^2)
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)

theorem inequality_proof : a < c ∧ c < b := by
  sorry

end inequality_proof_l190_190107


namespace strawberry_harvest_l190_190991

theorem strawberry_harvest
  (length : ℕ) (width : ℕ)
  (plants_per_sqft : ℕ) (yield_per_plant : ℕ)
  (garden_area : ℕ := length * width) 
  (total_plants : ℕ := plants_per_sqft * garden_area) 
  (expected_strawberries : ℕ := yield_per_plant * total_plants) :
  length = 10 ∧ width = 12 ∧ plants_per_sqft = 5 ∧ yield_per_plant = 8 → 
  expected_strawberries = 4800 := by
  sorry

end strawberry_harvest_l190_190991


namespace total_kids_in_lawrence_county_l190_190459

def kids_stayed_home : ℕ := 644997
def kids_went_to_camp : ℕ := 893835
def kids_from_outside : ℕ := 78

theorem total_kids_in_lawrence_county : kids_stayed_home + kids_went_to_camp = 1538832 := by
  sorry

end total_kids_in_lawrence_county_l190_190459


namespace expected_value_winnings_l190_190818

def probability_heads : ℚ := 2 / 5
def probability_tails : ℚ := 3 / 5
def winnings_heads : ℚ := 4
def loss_tails : ℚ := -3

theorem expected_value_winnings : 
  (probability_heads * winnings_heads + probability_tails * loss_tails) = -1 / 5 := 
by
  -- calculation steps and proof would go here
  sorry

end expected_value_winnings_l190_190818


namespace distance_to_right_focus_l190_190255

variable (F1 F2 P : ℝ × ℝ)
variable (a : ℝ)
variable (h_ellipse : ∀ P : ℝ × ℝ, P ∈ { P : ℝ × ℝ | (P.1^2 / 9) + (P.2^2 / 8) = 1 })
variable (h_foci_dist : (P : ℝ × ℝ) → (F1 : ℝ × ℝ) → (F2 : ℝ × ℝ) → (dist P F1) = 2)
variable (semi_major_axis : a = 3)

theorem distance_to_right_focus (h : dist F1 F2 = 2 * a) : dist P F2 = 4 := 
sorry

end distance_to_right_focus_l190_190255


namespace eval_expression_l190_190396

theorem eval_expression : (49^2 - 25^2 + 10^2) = 1876 := by
  sorry

end eval_expression_l190_190396


namespace point_B_value_l190_190918

/-- Given that point A represents the number 7 on a number line
    and point A is moved 3 units to the right to point B,
    prove that point B represents the number 10 -/
theorem point_B_value (A B : ℤ) (h1: A = 7) (h2: B = A + 3) : B = 10 :=
  sorry

end point_B_value_l190_190918


namespace miniature_tower_height_l190_190658

theorem miniature_tower_height
  (actual_height : ℝ)
  (actual_volume : ℝ)
  (miniature_volume : ℝ)
  (actual_height_eq : actual_height = 60)
  (actual_volume_eq : actual_volume = 200000)
  (miniature_volume_eq : miniature_volume = 0.2) :
  ∃ (miniature_height : ℝ), miniature_height = 0.6 :=
by
  sorry

end miniature_tower_height_l190_190658


namespace max_value_M_l190_190325

def J_k (k : ℕ) : ℕ := 10^(k + 3) + 1600

def M (k : ℕ) : ℕ := (J_k k).factors.count 2

theorem max_value_M : ∃ k > 0, (M k) = 7 ∧ ∀ m > 0, M m ≤ 7 :=
by 
  sorry

end max_value_M_l190_190325


namespace fourth_term_row1_is_16_nth_term_row1_nth_term_row2_sum_three_consecutive_row3_l190_190718

-- Define the sequences as functions
def row1 (n : ℕ) : ℤ := (-2)^n
def row2 (n : ℕ) : ℤ := row1 n + 2
def row3 (n : ℕ) : ℤ := (-1) * (-2)^n

-- Theorems to be proven

-- (1) Prove the fourth term in row ① is 16 
theorem fourth_term_row1_is_16 : row1 4 = 16 := sorry

-- (1) Prove the nth term in row ① is (-2)^n
theorem nth_term_row1 (n : ℕ) : row1 n = (-2)^n := sorry

-- (2) Let the nth number in row ① be a, prove the nth number in row ② is a + 2
theorem nth_term_row2 (n : ℕ) : row2 n = row1 n + 2 := sorry

-- (3) If the sum of three consecutive numbers in row ③ is -192, find these numbers
theorem sum_three_consecutive_row3 : ∃ n : ℕ, row3 n + row3 (n + 1) + row3 (n + 2) = -192 ∧ 
  row3 n  = -64 ∧ row3 (n + 1) = 128 ∧ row3 (n + 2) = -256 := sorry

end fourth_term_row1_is_16_nth_term_row1_nth_term_row2_sum_three_consecutive_row3_l190_190718


namespace blending_marker_drawings_correct_l190_190302

-- Define the conditions
def total_drawings : ℕ := 25
def colored_pencil_drawings : ℕ := 14
def charcoal_drawings : ℕ := 4

-- Define the target proof statement
def blending_marker_drawings : ℕ := total_drawings - (colored_pencil_drawings + charcoal_drawings)

-- Proof goal
theorem blending_marker_drawings_correct : blending_marker_drawings = 7 := by
  sorry

end blending_marker_drawings_correct_l190_190302


namespace total_cost_proof_l190_190792

-- Definitions for the problem conditions
def basketball_cost : ℕ := 48
def volleyball_cost : ℕ := basketball_cost - 18
def basketball_quantity : ℕ := 3
def volleyball_quantity : ℕ := 5
def total_basketball_cost : ℕ := basketball_cost * basketball_quantity
def total_volleyball_cost : ℕ := volleyball_cost * volleyball_quantity
def total_cost : ℕ := total_basketball_cost + total_volleyball_cost

-- Theorem to be proved
theorem total_cost_proof : total_cost = 294 :=
by
  sorry

end total_cost_proof_l190_190792


namespace truck_cargo_solution_l190_190364

def truck_cargo_problem (x : ℝ) (n : ℕ) : Prop :=
  (∀ (x : ℝ) (n : ℕ), x = (x / n - 0.5) * (n + 4)) ∧ (55 ≤ x ∧ x ≤ 64)

theorem truck_cargo_solution :
  ∃ y : ℝ, y = 2.5 :=
sorry

end truck_cargo_solution_l190_190364


namespace sum_r_p_values_l190_190602

def p (x : ℝ) : ℝ := |x| - 2
def r (x : ℝ) : ℝ := -|p x - 1|
def r_p (x : ℝ) : ℝ := r (p x)

theorem sum_r_p_values :
  (r_p (-4) + r_p (-3) + r_p (-2) + r_p (-1) + r_p 0 + r_p 1 + r_p 2 + r_p 3 + r_p 4) = -11 :=
by 
  -- Proof omitted
  sorry

end sum_r_p_values_l190_190602


namespace fraction_subtraction_l190_190346

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem fraction_subtraction : 
  (18 / 42 - 2 / 9) = (13 / 63) := 
by 
  sorry

end fraction_subtraction_l190_190346


namespace correct_statement_exam_l190_190127

theorem correct_statement_exam 
  (students_participated : ℕ)
  (students_sampled : ℕ)
  (statement1 : Bool)
  (statement2 : Bool)
  (statement3 : Bool)
  (statement4 : Bool)
  (cond1 : students_participated = 70000)
  (cond2 : students_sampled = 1000)
  (cond3 : statement1 = False)
  (cond4 : statement2 = False)
  (cond5 : statement3 = False)
  (cond6 : statement4 = True) :
  statement4 = True := 
sorry

end correct_statement_exam_l190_190127


namespace integer_solution_existence_l190_190621

theorem integer_solution_existence : ∃ (x y : ℤ), 2 * x + y - 1 = 0 :=
by
  use 1
  use -1
  sorry

end integer_solution_existence_l190_190621


namespace tip_percentage_l190_190672

theorem tip_percentage
  (total_amount_paid : ℝ)
  (price_of_food : ℝ)
  (sales_tax_rate : ℝ)
  (total_amount : ℝ)
  (tip_percentage : ℝ)
  (h1 : total_amount_paid = 184.80)
  (h2 : price_of_food = 140)
  (h3 : sales_tax_rate = 0.10)
  (h4 : total_amount = price_of_food + (price_of_food * sales_tax_rate))
  (h5 : tip_percentage = ((total_amount_paid - total_amount) / total_amount) * 100) :
  tip_percentage = 20 := sorry

end tip_percentage_l190_190672


namespace intersection_sums_l190_190467

def parabola1 (x : ℝ) : ℝ := (x - 2)^2
def parabola2 (y : ℝ) : ℝ := (y - 2)^2 - 6

theorem intersection_sums (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) 
  (h1 : y1 = parabola1 x1) (h2 : y2 = parabola1 x2)
  (h3 : y3 = parabola1 x3) (h4 : y4 = parabola1 x4)
  (k1 : x1 + 6 = y1^2 - 4*y1 + 4) (k2 : x2 + 6 = y2^2 - 4*y2 + 4)
  (k3 : x3 + 6 = y3^2 - 4*y3 + 4) (k4 : x4 + 6 = y4^2 - 4*y4 + 4) :
  x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 16 := 
sorry

end intersection_sums_l190_190467


namespace angle_between_clock_hands_at_3_05_l190_190853

theorem angle_between_clock_hands_at_3_05 :
  let minute_angle := 5 * 6
  let hour_angle := (5 / 60) * 30
  let initial_angle := 3 * 30
  initial_angle - minute_angle + hour_angle = 62.5 := by
  sorry

end angle_between_clock_hands_at_3_05_l190_190853


namespace total_consumer_installment_credit_l190_190955

-- Conditions
def auto_instalment_credit (C : ℝ) : ℝ := 0.2 * C
def auto_finance_extends_1_third (auto_installment : ℝ) : ℝ := 57
def student_loans (C : ℝ) : ℝ := 0.15 * C
def credit_card_debt (C : ℝ) (auto_installment : ℝ) : ℝ := 0.25 * C
def other_loans (C : ℝ) : ℝ := 0.4 * C

-- Correct Answer
theorem total_consumer_installment_credit (C : ℝ) :
  auto_instalment_credit C / 3 = auto_finance_extends_1_third (auto_instalment_credit C) ∧
  student_loans C = 80 ∧
  credit_card_debt C (auto_instalment_credit C) = auto_instalment_credit C + 100 ∧
  credit_card_debt C (auto_instalment_credit C) = 271 →
  C = 1084 := 
by
  sorry

end total_consumer_installment_credit_l190_190955


namespace initial_markup_percentage_l190_190340

theorem initial_markup_percentage (C : ℝ) (M : ℝ) :
  (C > 0) →
  (1 + M) * 1.25 * 0.90 = 1.35 →
  M = 0.2 :=
by
  intros
  sorry

end initial_markup_percentage_l190_190340


namespace sum_abs_a1_to_a10_l190_190385

def S (n : ℕ) : ℤ := n^2 - 4 * n + 2
def a (n : ℕ) : ℤ := if n = 1 then S 1 else S n - S (n - 1)

theorem sum_abs_a1_to_a10 : (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 66) := 
by
  sorry

end sum_abs_a1_to_a10_l190_190385


namespace find_essay_pages_l190_190131

/-
Conditions:
1. It costs $0.10 to print one page.
2. Jenny wants to print 7 copies of her essay.
3. Jenny wants to buy 7 pens that each cost $1.50.
4. Jenny pays the store with 2 twenty dollar bills and gets $12 in change.
-/

def cost_per_page : Float := 0.10
def number_of_copies : Nat := 7
def cost_per_pen : Float := 1.50
def number_of_pens : Nat := 7
def total_money_given : Float := 40.00  -- 2 twenty dollar bills
def change_received : Float := 12.00

theorem find_essay_pages :
  let total_spent := total_money_given - change_received
  let total_cost_of_pens := Float.ofNat number_of_pens * cost_per_pen
  let total_amount_spent_on_printing := total_spent - total_cost_of_pens
  let number_of_pages := total_amount_spent_on_printing / cost_per_page
  number_of_pages = 175 := by
  sorry

end find_essay_pages_l190_190131


namespace dusty_change_l190_190736

def price_single_layer : ℕ := 4
def price_double_layer : ℕ := 7
def number_of_single_layers : ℕ := 7
def number_of_double_layers : ℕ := 5
def amount_paid : ℕ := 100

theorem dusty_change :
  amount_paid - (number_of_single_layers * price_single_layer + number_of_double_layers * price_double_layer) = 37 := 
by
  sorry

end dusty_change_l190_190736


namespace smallest_number_divisible_l190_190910

   theorem smallest_number_divisible (d n : ℕ) (h₁ : (n + 7) % 11 = 0) (h₂ : (n + 7) % 24 = 0) (h₃ : (n + 7) % d = 0) (h₄ : (n + 7) = 257) : n = 250 :=
   by
     sorry
   
end smallest_number_divisible_l190_190910


namespace solve_inequality_l190_190164

theorem solve_inequality : { x : ℝ // (x < -1) ∨ (-2/3 < x) } :=
sorry

end solve_inequality_l190_190164


namespace binary_to_base5_conversion_l190_190628

theorem binary_to_base5_conversion : ∀ (b : ℕ), b = 1101 → (13 : ℕ) % 5 = 3 ∧ (13 / 5) % 5 = 2 → b = 1101 → (1101 : ℕ) = 13 → 13 = 23 :=
by
  sorry

end binary_to_base5_conversion_l190_190628


namespace working_mom_hours_at_work_l190_190940

-- Definitions corresponding to the conditions
def hours_awake : ℕ := 16
def work_percentage : ℝ := 0.50

-- The theorem to be proved
theorem working_mom_hours_at_work : work_percentage * hours_awake = 8 :=
by sorry

end working_mom_hours_at_work_l190_190940


namespace mark_buttons_l190_190254

/-- Mark started the day with some buttons. His friend Shane gave him 3 times that amount of buttons.
    Then his other friend Sam asked if he could have half of Mark’s buttons. 
    Mark ended up with 28 buttons. How many buttons did Mark start the day with? --/
theorem mark_buttons (B : ℕ) (h1 : 2 * B = 28) : B = 14 := by
  sorry

end mark_buttons_l190_190254


namespace max_marks_l190_190151

theorem max_marks (M S : ℕ) :
  (267 + 45 = 312) ∧ (312 = (45 * M) / 100) ∧ (292 + 38 = 330) ∧ (330 = (50 * S) / 100) →
  (M + S = 1354) :=
by
  sorry

end max_marks_l190_190151


namespace find_largest_x_and_compute_ratio_l190_190155

theorem find_largest_x_and_compute_ratio (a b c d : ℤ) (h : x = (a + b * Real.sqrt c) / d)
   (cond : (5 * x / 7) + 1 = 3 / x) : a * c * d / b = -70 :=
by
  sorry

end find_largest_x_and_compute_ratio_l190_190155


namespace ratio_of_surface_areas_l190_190946

-- Definitions based on conditions
def side_length_ratio (a b : ℝ) : Prop := b = 6 * a
def surface_area (a : ℝ) : ℝ := 6 * a ^ 2

-- Theorem statement
theorem ratio_of_surface_areas (a b : ℝ) (h : side_length_ratio a b) :
  (surface_area b) / (surface_area a) = 36 := by
  sorry

end ratio_of_surface_areas_l190_190946


namespace original_price_of_cycle_l190_190867

variable (P : ℝ)

theorem original_price_of_cycle (h1 : 0.75 * P = 1050) : P = 1400 :=
sorry

end original_price_of_cycle_l190_190867


namespace caesars_charge_l190_190235

theorem caesars_charge :
  ∃ (C : ℕ), (C + 30 * 60 = 500 + 35 * 60) ↔ (C = 800) :=
by
  sorry

end caesars_charge_l190_190235


namespace ab_value_l190_190152

theorem ab_value (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 29) : a * b = 2 :=
by
  -- proof will be provided here
  sorry

end ab_value_l190_190152


namespace correct_option_d_l190_190520

-- Define the conditions as separate lemmas
lemma option_a_incorrect : ¬ (Real.sqrt 18 + Real.sqrt 2 = 2 * Real.sqrt 5) :=
sorry 

lemma option_b_incorrect : ¬ (Real.sqrt 18 - Real.sqrt 2 = 4) :=
sorry

lemma option_c_incorrect : ¬ (Real.sqrt 18 * Real.sqrt 2 = 36) :=
sorry

-- Define the statement to prove
theorem correct_option_d : Real.sqrt 18 / Real.sqrt 2 = 3 :=
by
  sorry

end correct_option_d_l190_190520


namespace initial_height_after_10_seconds_l190_190726

open Nat

def distance_fallen_in_nth_second (n : ℕ) : ℕ := 10 * n - 5

def total_distance_fallen (n : ℕ) : ℕ :=
  (n * (distance_fallen_in_nth_second 1 + distance_fallen_in_nth_second n)) / 2

theorem initial_height_after_10_seconds : 
  total_distance_fallen 10 = 500 := 
by
  sorry

end initial_height_after_10_seconds_l190_190726


namespace sufficient_condition_range_a_l190_190007

theorem sufficient_condition_range_a (a : ℝ) :
  (∀ x, (2 * a ≤ x ∧ x ≤ a^2 + 1) → (x^2 - 3 * (a + 1) * x + 6 * a + 2 ≤ 0)) ↔
  (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) := by
  sorry

end sufficient_condition_range_a_l190_190007


namespace find_certain_number_l190_190836

noncomputable def certain_number_is_square (n : ℕ) (x : ℕ) : Prop :=
  ∃ (y : ℕ), x * n = y * y

theorem find_certain_number : ∃ x, certain_number_is_square 3 x :=
by 
  use 1
  unfold certain_number_is_square
  use 3
  sorry

end find_certain_number_l190_190836


namespace total_letters_sent_l190_190236

-- Define the number of letters sent in each month
def letters_in_January : ℕ := 6
def letters_in_February : ℕ := 9
def letters_in_March : ℕ := 3 * letters_in_January

-- Theorem statement: the total number of letters sent across the three months
theorem total_letters_sent :
  letters_in_January + letters_in_February + letters_in_March = 33 := by
  sorry

end total_letters_sent_l190_190236


namespace arriving_late_l190_190488

-- Definitions from conditions
def usual_time : ℕ := 24
def slower_factor : ℚ := 3 / 4

-- Derived from conditions
def slower_time : ℚ := usual_time * (4 / 3)

-- To be proven
theorem arriving_late : slower_time - usual_time = 8 := by
  sorry

end arriving_late_l190_190488


namespace lucy_total_fish_l190_190165

variable (current_fish additional_fish : ℕ)

def total_fish (current_fish additional_fish : ℕ) : ℕ :=
  current_fish + additional_fish

theorem lucy_total_fish (h1 : current_fish = 212) (h2 : additional_fish = 68) : total_fish current_fish additional_fish = 280 :=
by
  sorry

end lucy_total_fish_l190_190165


namespace slope_acute_l190_190889

noncomputable def curve (a : ℤ) : ℝ → ℝ := λ x => x^3 - 2 * a * x^2 + 2 * a * x

noncomputable def tangent_slope (a : ℤ) : ℝ → ℝ := λ x => 3 * x^2 - 4 * a * x + 2 * a

theorem slope_acute (a : ℤ) : (∀ x : ℝ, (tangent_slope a x > 0)) ↔ (a = 1) := sorry

end slope_acute_l190_190889


namespace area_relation_l190_190802

-- Define the areas of the triangles
variables (a b c : ℝ)

-- Define the condition that triangles T_a and T_c are similar (i.e., homothetic)
-- which implies the relationship between their areas.
theorem area_relation (ha : 0 < a) (hc : 0 < c) (habc : b = Real.sqrt (a * c)) : b = Real.sqrt (a * c) := by
  sorry

end area_relation_l190_190802


namespace find_x_eq_2_l190_190482

theorem find_x_eq_2 (x : ℕ) (h : 7899665 - 36 * x = 7899593) : x = 2 := 
by
  sorry

end find_x_eq_2_l190_190482


namespace arithmetic_sequence_difference_l190_190511

noncomputable def arithmetic_difference (d: ℚ) (b₁: ℚ) : Prop :=
  (50 * b₁ + ((50 * 49) / 2) * d = 150) ∧
  (50 * (b₁ + 50 * d) + ((50 * 149) / 2) * d = 250)

theorem arithmetic_sequence_difference {d b₁ : ℚ} (h : arithmetic_difference d b₁) :
  (b₁ + d) - b₁ = (200 / 1295) :=
by
  sorry

end arithmetic_sequence_difference_l190_190511


namespace sum_of_squares_eq_l190_190978

theorem sum_of_squares_eq :
  (1000^2 + 1001^2 + 1002^2 + 1003^2 + 1004^2 + 1005^2 + 1006^2) = 7042091 :=
by {
  sorry
}

end sum_of_squares_eq_l190_190978


namespace cut_into_four_and_reassemble_l190_190990

-- Definitions as per conditions in the problem
def figureArea : ℕ := 36
def nParts : ℕ := 4
def squareArea (s : ℕ) : ℕ := s * s

-- Property to be proved
theorem cut_into_four_and_reassemble :
  ∃ (s : ℕ), squareArea s = figureArea / nParts ∧ s * s = figureArea :=
by
  sorry

end cut_into_four_and_reassemble_l190_190990


namespace miles_traveled_total_l190_190443

-- Define the initial distance and the additional distance
def initial_distance : ℝ := 212.3
def additional_distance : ℝ := 372.0

-- Define the total distance as the sum of the initial and additional distances
def total_distance : ℝ := initial_distance + additional_distance

-- Prove that the total distance is 584.3 miles
theorem miles_traveled_total : total_distance = 584.3 := by
  sorry

end miles_traveled_total_l190_190443


namespace factorization_correct_l190_190757

theorem factorization_correct : ∃ (a b : ℕ), (a > b) ∧ (3 * b - a = 12) ∧ (x^2 - 16 * x + 63 = (x - a) * (x - b)) :=
by
  sorry

end factorization_correct_l190_190757


namespace min_value_of_even_function_l190_190476

-- Define f(x) = (x + a)(x + b)
def f (x a b : ℝ) : ℝ := (x + a) * (x + b)

-- Given conditions
variables (a b : ℝ)
#check f  -- Ensuring the definition works

-- Prove that the minimum value of f(x) is -4 given that f(x) is an even function
theorem min_value_of_even_function (h_even : ∀ x : ℝ, f x a b = f (-x) a b)
  (h_domain : a + 4 > a) : ∃ c : ℝ, (f c a b = -4) :=
by
  -- We state that this function is even and consider the provided domain.
  sorry  -- Placeholder for the proof

end min_value_of_even_function_l190_190476


namespace smallest_common_multiple_l190_190552

theorem smallest_common_multiple (n : ℕ) : 
  (2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n ∧ 1000 ≤ n ∧ n < 10000) → n = 1008 :=
by {
    sorry
}

end smallest_common_multiple_l190_190552


namespace students_problem_count_l190_190422

theorem students_problem_count 
  (x y z q r : ℕ) 
  (H1 : x + y + z + q + r = 30) 
  (H2 : x + 2 * y + 3 * z + 4 * q + 5 * r = 40) 
  (h_y_pos : 1 ≤ y) 
  (h_z_pos : 1 ≤ z) 
  (h_q_pos : 1 ≤ q) 
  (h_r_pos : 1 ≤ r) : 
  x = 26 := 
  sorry

end students_problem_count_l190_190422


namespace find_smallest_c_l190_190708

theorem find_smallest_c (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
    (graph_eq : ∀ x, (a * Real.sin (b * x + c) + d) = 5 → x = (π / 6))
    (amplitude_eq : a = 3) : c = π / 2 :=
sorry

end find_smallest_c_l190_190708


namespace find_m_minus_n_l190_190697

theorem find_m_minus_n (x y m n : ℤ) (h1 : x = -2) (h2 : y = 1) 
  (h3 : 3 * x + 2 * y = m) (h4 : n * x - y = 1) : m - n = -3 :=
by sorry

end find_m_minus_n_l190_190697


namespace initial_card_count_l190_190839

theorem initial_card_count (x : ℕ) (h1 : (3 * (1/2) * ((x / 3) + (4 / 3))) = 34) : x = 64 :=
  sorry

end initial_card_count_l190_190839


namespace at_least_two_pass_written_test_expectation_number_of_admission_advantage_l190_190081

noncomputable def probability_of_passing_written_test_A : ℝ := 0.4
noncomputable def probability_of_passing_written_test_B : ℝ := 0.8
noncomputable def probability_of_passing_written_test_C : ℝ := 0.5

noncomputable def probability_of_passing_interview_A : ℝ := 0.8
noncomputable def probability_of_passing_interview_B : ℝ := 0.4
noncomputable def probability_of_passing_interview_C : ℝ := 0.64

theorem at_least_two_pass_written_test :
  (probability_of_passing_written_test_A * probability_of_passing_written_test_B * (1 - probability_of_passing_written_test_C) +
  probability_of_passing_written_test_A * (1 - probability_of_passing_written_test_B) * probability_of_passing_written_test_C +
  (1 - probability_of_passing_written_test_A) * probability_of_passing_written_test_B * probability_of_passing_written_test_C +
  probability_of_passing_written_test_A * probability_of_passing_written_test_B * probability_of_passing_written_test_C = 0.6) :=
sorry

theorem expectation_number_of_admission_advantage :
  (3 * (probability_of_passing_written_test_A * probability_of_passing_interview_A) +
  3 * (probability_of_passing_written_test_B * probability_of_passing_interview_B) +
  3 * (probability_of_passing_written_test_C * probability_of_passing_interview_C) = 0.96) :=
sorry

end at_least_two_pass_written_test_expectation_number_of_admission_advantage_l190_190081


namespace position_after_2010_transformations_l190_190030

-- Define the initial position of the square
def init_position := "ABCD"

-- Define the transformation function
def transform (position : String) (steps : Nat) : String :=
  match steps % 8 with
  | 0 => "ABCD"
  | 1 => "CABD"
  | 2 => "DACB"
  | 3 => "BCAD"
  | 4 => "ADCB"
  | 5 => "CBDA"
  | 6 => "BADC"
  | 7 => "CDAB"
  | _ => "ABCD"  -- Default case, should never happen

-- The theorem to prove the correct position after 2010 transformations
theorem position_after_2010_transformations : transform init_position 2010 = "CABD" := 
by
  sorry

end position_after_2010_transformations_l190_190030


namespace find_p_l190_190732

theorem find_p (m n p : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < p) 
  (h : 3 * m + 3 / (n + 1 / p) = 17) : p = 2 := 
sorry

end find_p_l190_190732


namespace triangle_side_AC_value_l190_190589

theorem triangle_side_AC_value
  (AB BC : ℝ) (AC : ℕ)
  (hAB : AB = 1)
  (hBC : BC = 2007)
  (hAC_int : ∃ (n : ℕ), AC = n) :
  AC = 2007 :=
by
  sorry

end triangle_side_AC_value_l190_190589


namespace polynomial_roots_property_l190_190362

theorem polynomial_roots_property (a b : ℝ) (h : ∀ x, x^2 + x - 2024 = 0 → x = a ∨ x = b) : 
  a^2 + 2 * a + b = 2023 :=
by
  sorry

end polynomial_roots_property_l190_190362


namespace total_balloons_cost_is_91_l190_190813

-- Define the number of balloons and their costs for Fred, Sam, and Dan
def fred_balloons : ℕ := 10
def fred_cost_per_balloon : ℝ := 1

def sam_balloons : ℕ := 46
def sam_cost_per_balloon : ℝ := 1.5

def dan_balloons : ℕ := 16
def dan_cost_per_balloon : ℝ := 0.75

-- Calculate the total cost for each person’s balloons
def fred_total_cost : ℝ := fred_balloons * fred_cost_per_balloon
def sam_total_cost : ℝ := sam_balloons * sam_cost_per_balloon
def dan_total_cost : ℝ := dan_balloons * dan_cost_per_balloon

-- Calculate the total cost of all the balloons combined
def total_cost : ℝ := fred_total_cost + sam_total_cost + dan_total_cost

-- The main statement to be proved
theorem total_balloons_cost_is_91 : total_cost = 91 :=
by
  -- Recall that the previous individual costs can be worked out and added
  -- But for the sake of this statement, we use sorry to skip details
  sorry

end total_balloons_cost_is_91_l190_190813


namespace find_b_l190_190193

noncomputable def func (x a b : ℝ) := (1 / 12) * x^2 + a * x + b

theorem  find_b (a b : ℝ) (x1 x2 : ℝ):
    (func x1 a b = 0) →
    (func x2 a b = 0) →
    (b = (x1 * x2) / 12) →
    ((3 - x1) = (x2 - 3)) →
    (b = -6) :=
by
    sorry

end find_b_l190_190193


namespace exists_positive_m_f99_divisible_1997_l190_190282

def f (x : ℕ) : ℕ := 3 * x + 2

noncomputable
def higher_order_f (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => sorry  -- placeholder since f^0 isn't defined in this context
  | 1 => f x
  | k + 1 => f (higher_order_f k x)

theorem exists_positive_m_f99_divisible_1997 :
  ∃ m : ℕ, m > 0 ∧ higher_order_f 99 m % 1997 = 0 :=
sorry

end exists_positive_m_f99_divisible_1997_l190_190282


namespace intersection_P_Q_l190_190065

def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = -x + 2}

theorem intersection_P_Q : P ∩ Q = {y | y ≤ 2} :=
sorry

end intersection_P_Q_l190_190065


namespace triangle_angle_side_cases_l190_190654

theorem triangle_angle_side_cases
  (b c : ℝ) (B : ℝ)
  (hb : b = 3)
  (hc : c = 3 * Real.sqrt 3)
  (hB : B = Real.pi / 6) :
  (∃ A C a, A = Real.pi / 2 ∧ C = Real.pi / 3 ∧ a = Real.sqrt 21) ∨
  (∃ A C a, A = Real.pi / 6 ∧ C = 2 * Real.pi / 3 ∧ a = 3) :=
by
  sorry

end triangle_angle_side_cases_l190_190654


namespace sequence_inequality_l190_190924

def F : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n+2) => F (n+1) + F n

theorem sequence_inequality (n : ℕ) :
  (F (n+1) : ℝ)^(1 / n) ≥ 1 + 1 / ((F n : ℝ)^(1 / n)) :=
by
  sorry

end sequence_inequality_l190_190924


namespace distance_A_to_B_is_7km_l190_190943

theorem distance_A_to_B_is_7km
  (v1 v2 : ℝ) 
  (t_meet_before : ℝ)
  (t1_after_meet t2_after_meet : ℝ)
  (d1_before_meet d2_before_meet : ℝ)
  (d_after_meet : ℝ)
  (h1 : d1_before_meet = d2_before_meet + 1)
  (h2 : t_meet_before = d1_before_meet / v1)
  (h3 : t_meet_before = d2_before_meet / v2)
  (h4 : t1_after_meet = 3 / 4)
  (h5 : t2_after_meet = 4 / 3)
  (h6 : d1_before_meet + v1 * t1_after_meet = d_after_meet)
  (h7 : d2_before_meet + v2 * t2_after_meet = d_after_meet)
  : d_after_meet = 7 := 
sorry

end distance_A_to_B_is_7km_l190_190943


namespace gear_angular_speeds_ratio_l190_190799

noncomputable def gear_ratio (x y z w : ℕ) (ω_A ω_B ω_C ω_D : ℝ) :=
  x * ω_A = y * ω_B ∧ y * ω_B = z * ω_C ∧ z * ω_C = w * ω_D

theorem gear_angular_speeds_ratio (x y z w : ℕ) (ω_A ω_B ω_C ω_D : ℝ) 
  (h : gear_ratio x y z w ω_A ω_B ω_C ω_D) :
  ω_A / ω_B = y / x ∧ ω_B / ω_C = z / y ∧ ω_C / ω_D = w / z :=
by sorry

end gear_angular_speeds_ratio_l190_190799


namespace mango_distribution_l190_190964

theorem mango_distribution (harvested_mangoes : ℕ) (sold_fraction : ℕ) (received_per_neighbor : ℕ)
  (h_harvested : harvested_mangoes = 560)
  (h_sold_fraction : sold_fraction = 2)
  (h_received_per_neighbor : received_per_neighbor = 35) :
  (harvested_mangoes / sold_fraction) = (harvested_mangoes / sold_fraction) / received_per_neighbor :=
by
  sorry

end mango_distribution_l190_190964


namespace squirrel_journey_time_l190_190072

theorem squirrel_journey_time : 
  (let distance := 2
  let speed_to_tree := 3
  let speed_return := 2
  let time_to_tree := distance / speed_to_tree
  let time_return := distance / speed_return
  let total_time := (time_to_tree + time_return) * 60
  total_time = 100) :=
by
  sorry

end squirrel_journey_time_l190_190072


namespace part1_part2_l190_190895

def p (x a : ℝ) : Prop := x - a < 0
def q (x : ℝ) : Prop := x * x - 4 * x + 3 ≤ 0

theorem part1 (a : ℝ) (h : a = 2) (hpq : ∀ x : ℝ, p x a ∧ q x) :
  Set.Ico 1 (2 : ℝ) = {x : ℝ | p x a ∧ q x} :=
by {
  sorry
}

theorem part2 (hp : ∀ (x a : ℝ), p x a → ¬ q x) : {a : ℝ | ∀ x : ℝ, q x → p x a} = Set.Ioi 3 :=
by {
  sorry
}

end part1_part2_l190_190895


namespace Michael_pizza_fraction_l190_190243

theorem Michael_pizza_fraction (T : ℚ) (L : ℚ) (total : ℚ) (M : ℚ) 
  (hT : T = 1 / 2) (hL : L = 1 / 6) (htotal : total = 1) (hM : total - (T + L) = M) :
  M = 1 / 3 := 
sorry

end Michael_pizza_fraction_l190_190243


namespace problem_value_eq_13_l190_190287

theorem problem_value_eq_13 : 8 / 4 - 3^2 + 4 * 5 = 13 :=
by
  sorry

end problem_value_eq_13_l190_190287


namespace range_of_m_for_nonnegative_quadratic_l190_190588

-- The statement of the proof problem in Lean
theorem range_of_m_for_nonnegative_quadratic {x m : ℝ} : 
  (∀ x, x^2 + m*x + 1 ≥ 0) ↔ -2 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_for_nonnegative_quadratic_l190_190588


namespace complex_fraction_sum_real_parts_l190_190445

theorem complex_fraction_sum_real_parts (a b : ℝ) (h : (⟨0, 1⟩ / ⟨1, 1⟩ : ℂ) = a + b * ⟨0, 1⟩) : a + b = 1 := by
  sorry

end complex_fraction_sum_real_parts_l190_190445


namespace probability_of_square_or_circle_is_seven_tenths_l190_190607

-- Define the total number of figures
def total_figures : ℕ := 10

-- Define the number of squares
def num_squares : ℕ := 4

-- Define the number of circles
def num_circles : ℕ := 3

-- The number of squares or circles
def num_squares_or_circles : ℕ := num_squares + num_circles

-- The probability of selecting a square or a circle
def probability_square_or_circle : ℚ := num_squares_or_circles / total_figures

-- The theorem stating the required proof
theorem probability_of_square_or_circle_is_seven_tenths :
  probability_square_or_circle = 7/10 :=
sorry -- proof goes here

end probability_of_square_or_circle_is_seven_tenths_l190_190607


namespace find_y_solution_l190_190673

variable (y : ℚ)

theorem find_y_solution (h : (y^2 - 12*y + 32) / (y - 2) + (3*y^2 + 11*y - 14) / (3*y - 1) = -5) : 
    y = -17/6 :=
by
  sorry

end find_y_solution_l190_190673


namespace points_comparison_l190_190327

def quadratic_function (m x : ℝ) : ℝ :=
  (x + m - 3) * (x - m) + 3

def point_on_graph (m x y : ℝ) : Prop :=
  y = quadratic_function m x

theorem points_comparison (m x1 x2 y1 y2 : ℝ)
  (h1 : point_on_graph m x1 y1)
  (h2 : point_on_graph m x2 y2)
  (hx : x1 < x2)
  (h_sum : x1 + x2 < 3) :
  y1 > y2 := 
  sorry

end points_comparison_l190_190327


namespace talias_fathers_age_l190_190038

-- Definitions based on the conditions
variable (T M F : ℕ)

-- The conditions
axiom h1 : T + 7 = 20
axiom h2 : M = 3 * T
axiom h3 : F + 3 = M

-- Goal: Prove that Talia's father (F) is currently 36 years old
theorem talias_fathers_age : F = 36 :=
by
  sorry

end talias_fathers_age_l190_190038


namespace bananas_left_l190_190624

theorem bananas_left (dozen_bananas : ℕ) (eaten_bananas : ℕ) (h1 : dozen_bananas = 12) (h2 : eaten_bananas = 2) : dozen_bananas - eaten_bananas = 10 :=
sorry

end bananas_left_l190_190624


namespace linear_regression_equation_l190_190380

theorem linear_regression_equation (x y : ℝ) (h : {(1, 2), (2, 3), (3, 4), (4, 5)} ⊆ {(x, y) | y = x + 1}) : 
  (∀ x y, (x = 1 → y = 2) ∧ (x = 2 → y = 3) ∧ (x = 3 → y = 4) ∧ (x = 4 → y = 5)) ↔ (y = x + 1) :=
by
  sorry

end linear_regression_equation_l190_190380


namespace range_of_a_for_two_critical_points_l190_190950

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - Real.exp 1 * x^2 + 18

theorem range_of_a_for_two_critical_points (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) ↔ (a ∈ Set.Ioo (1 / Real.exp 1) 1 ∪ Set.Ioo 1 (Real.exp 1)) :=
sorry

end range_of_a_for_two_critical_points_l190_190950


namespace tomato_count_after_harvest_l190_190024

theorem tomato_count_after_harvest :
  let plant_A_initial := 150
  let plant_B_initial := 200
  let plant_C_initial := 250
  -- Day 1
  let plant_A_after_day1 := plant_A_initial - (plant_A_initial * 3 / 10)
  let plant_B_after_day1 := plant_B_initial - (plant_B_initial * 1 / 4)
  let plant_C_after_day1 := plant_C_initial - (plant_C_initial * 4 / 25)
  -- Day 7
  let plant_A_after_day7 := plant_A_after_day1 - ((plant_A_initial * 3 / 10) + 5)
  let plant_B_after_day7 := plant_B_after_day1 - (plant_B_after_day1 * 1 / 5)
  let plant_C_after_day7 := plant_C_after_day1 - ((plant_C_initial * 4 / 25) * 2)
  -- Day 14
  let plant_A_after_day14 := plant_A_after_day7 - ((plant_A_after_day1 - ((plant_A_initial * 3 / 10) + 5)) * 3)
  let plant_B_after_day14 := plant_B_after_day7 - ((plant_B_after_day1 * 1 / 5) + 15)
  let plant_C_after_day14 := plant_C_after_day7 - (plant_C_after_day7 * 1 / 5)
  (plant_A_after_day14 = 0) ∧ (plant_B_after_day14 = 75) ∧ (plant_C_after_day14 = 104) :=
by
  sorry

end tomato_count_after_harvest_l190_190024


namespace part1_part2_l190_190750

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem part1 (x : ℝ) : (f x 2 ≥ 7 - |x - 1|) ↔ (x ≤ -2 ∨ x ≥ 5) :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x, f x a ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) : a = 1 :=
by sorry

end part1_part2_l190_190750


namespace probability_blue_is_4_over_13_l190_190495

def num_red : ℕ := 5
def num_green : ℕ := 6
def num_yellow : ℕ := 7
def num_blue : ℕ := 8
def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue

def probability_blue : ℚ := num_blue / total_jelly_beans

theorem probability_blue_is_4_over_13
  (h_num_red : num_red = 5)
  (h_num_green : num_green = 6)
  (h_num_yellow : num_yellow = 7)
  (h_num_blue : num_blue = 8) :
  probability_blue = 4 / 13 :=
by
  sorry

end probability_blue_is_4_over_13_l190_190495


namespace min_x1_x2_squared_l190_190186

theorem min_x1_x2_squared (x1 x2 m : ℝ) (hm : (m + 3)^2 ≥ 0) 
  (h_sum : x1 + x2 = -(m + 1)) 
  (h_prod : x1 * x2 = 2 * m - 2) : 
  (x1^2 + x2^2 = (m - 1)^2 + 4) ∧ ∃ m, m = 1 → x1^2 + x2^2 = 4 :=
by {
  sorry
}

end min_x1_x2_squared_l190_190186


namespace fishing_problem_l190_190472

theorem fishing_problem :
  ∃ F : ℕ, (F % 3 = 1 ∧
            ((F - 1) / 3) % 3 = 1 ∧
            ((((F - 1) / 3 - 1) / 3) % 3 = 1) ∧
            ((((F - 1) / 3 - 1) / 3 - 1) / 3) % 3 = 1 ∧
            ((((F - 1) / 3 - 1) / 3 - 1) / 3 - 1) = 0) :=
sorry

end fishing_problem_l190_190472


namespace sum_first_five_even_numbers_l190_190715

theorem sum_first_five_even_numbers : (2 + 4 + 6 + 8 + 10) = 30 :=
by
  sorry

end sum_first_five_even_numbers_l190_190715


namespace shift_graph_sin_cos_l190_190908

open Real

theorem shift_graph_sin_cos :
  ∀ x : ℝ, sin (2 * x + π / 3) = cos (2 * (x + π / 12) - π / 3) :=
by
  sorry

end shift_graph_sin_cos_l190_190908


namespace boat_distance_along_stream_l190_190226

-- Define the conditions
def boat_speed_still_water := 15 -- km/hr
def distance_against_stream_one_hour := 9 -- km

-- Define the speed of the stream
def stream_speed := boat_speed_still_water - distance_against_stream_one_hour -- km/hr

-- Define the effective speed along the stream
def effective_speed_along_stream := boat_speed_still_water + stream_speed -- km/hr

-- Define the proof statement
theorem boat_distance_along_stream : effective_speed_along_stream = 21 :=
by
  -- Given conditions and definitions, the steps are assumed logically correct
  sorry

end boat_distance_along_stream_l190_190226


namespace skateboard_total_distance_l190_190133

theorem skateboard_total_distance :
  let a_1 := 8
  let d := 6
  let n := 40
  let distance (m : ℕ) := a_1 + (m - 1) * d
  let S_n := n / 2 * (distance 1 + distance n)
  S_n = 5000 := by
  sorry

end skateboard_total_distance_l190_190133


namespace solution_f_derivative_l190_190551

noncomputable def f (x : ℝ) := Real.sqrt x

theorem solution_f_derivative :
  (deriv f 1) = 1 / 2 :=
by
  -- This is where the proof would go, but for now, we just state sorry.
  sorry

end solution_f_derivative_l190_190551


namespace volume_of_original_cube_l190_190956

theorem volume_of_original_cube (s : ℝ) (h : (s + 2) * (s - 3) * s - s^3 = 26) : s^3 = 343 := 
sorry

end volume_of_original_cube_l190_190956


namespace taishan_maiden_tea_prices_l190_190865

theorem taishan_maiden_tea_prices (x y : ℝ) 
  (h1 : 30 * x + 20 * y = 6000)
  (h2 : 24 * x + 18 * y = 5100) :
  x = 100 ∧ y = 150 :=
by
  sorry

end taishan_maiden_tea_prices_l190_190865


namespace function_neither_odd_nor_even_l190_190584

def f (x : ℝ) : ℝ := x^2 + 6 * x

theorem function_neither_odd_nor_even : 
  ¬ (∀ x, f (-x) = f x) ∧ ¬ (∀ x, f (-x) = -f x) :=
by
  sorry

end function_neither_odd_nor_even_l190_190584


namespace value_of_f_at_6_l190_190784

variable {R : Type*} [LinearOrderedField R]

noncomputable def f : R → R := sorry

-- Conditions
axiom odd_function (x : R) : f (-x) = -f x
axiom periodicity (x : R) : f (x + 2) = -f x

-- Theorem to prove
theorem value_of_f_at_6 : f 6 = 0 := by sorry

end value_of_f_at_6_l190_190784


namespace max_side_of_triangle_l190_190184

theorem max_side_of_triangle {a b c : ℕ} (h1: a + b + c = 24) (h2: a + b > c) (h3: a + c > b) (h4: b + c > a) :
  max a (max b c) = 11 :=
sorry

end max_side_of_triangle_l190_190184


namespace left_handed_and_like_scifi_count_l190_190505

-- Definitions based on the problem conditions
def total_members : ℕ := 30
def left_handed_members : ℕ := 12
def like_scifi_members : ℕ := 18
def right_handed_not_like_scifi : ℕ := 4

-- Main proof statement
theorem left_handed_and_like_scifi_count :
  ∃ x : ℕ, (left_handed_members - x) + (like_scifi_members - x) + x + right_handed_not_like_scifi = total_members ∧ x = 4 :=
by
  use 4
  sorry

end left_handed_and_like_scifi_count_l190_190505


namespace parabola_focus_segment_length_l190_190751

theorem parabola_focus_segment_length (a : ℝ) (h₀ : a > 0) 
  (h₁ : ∀ x, abs x * abs (1 / a) = 4) : a = 1/4 := 
sorry

end parabola_focus_segment_length_l190_190751


namespace min_small_bottles_needed_l190_190841

theorem min_small_bottles_needed (small_capacity large_capacity : ℕ) 
    (h_small_capacity : small_capacity = 35) (h_large_capacity : large_capacity = 500) : 
    ∃ n, n = 15 ∧ large_capacity <= n * small_capacity :=
by 
  sorry

end min_small_bottles_needed_l190_190841


namespace tan_double_angle_sub_l190_190583

theorem tan_double_angle_sub (α β : ℝ) 
  (h1 : Real.tan α = 1 / 2)
  (h2 : Real.tan (α - β) = 1 / 5) : Real.tan (2 * α - β) = 7 / 9 :=
by
  sorry

end tan_double_angle_sub_l190_190583


namespace choices_of_N_l190_190153

def base7_representation (N : ℕ) : ℕ := 
  (N / 49) * 100 + ((N % 49) / 7) * 10 + (N % 7)

def base8_representation (N : ℕ) : ℕ := 
  (N / 64) * 100 + ((N % 64) / 8) * 10 + (N % 8)

theorem choices_of_N : 
  ∃ (N_set : Finset ℕ), 
    (∀ N ∈ N_set, 100 ≤ N ∧ N < 1000 ∧ 
      ((base7_representation N * base8_representation N) % 100 = (3 * N) % 100)) 
    ∧ N_set.card = 15 :=
by
  sorry

end choices_of_N_l190_190153


namespace difference_of_squares_l190_190411

theorem difference_of_squares (x y : ℝ) (h₁ : x + y = 20) (h₂ : x - y = 10) : x^2 - y^2 = 200 :=
by {
  sorry
}

end difference_of_squares_l190_190411


namespace monomials_like_terms_l190_190307

theorem monomials_like_terms (a b : ℕ) (h1 : 3 = a) (h2 : 4 = 2 * b) : a = 3 ∧ b = 2 :=
by
  sorry

end monomials_like_terms_l190_190307


namespace find_fraction_l190_190834

-- Variables and Definitions
variables (x : ℚ)

-- Conditions
def condition1 := (2 / 3) / x = (3 / 5) / (7 / 15)

-- Theorem to prove the certain fraction
theorem find_fraction (h : condition1 x) : x = 14 / 27 :=
by sorry

end find_fraction_l190_190834


namespace abhay_speed_l190_190503

-- Definitions of the problem's conditions
def condition1 (A S : ℝ) : Prop := 42 / A = 42 / S + 2
def condition2 (A S : ℝ) : Prop := 42 / (2 * A) = 42 / S - 1

-- Define Abhay and Sameer's speeds and declare the main theorem
theorem abhay_speed (A S : ℝ) (h1 : condition1 A S) (h2 : condition2 A S) : A = 10.5 :=
by
  sorry

end abhay_speed_l190_190503


namespace cube_cut_possible_l190_190966

theorem cube_cut_possible (a b : ℝ) (unit_a : a = 1) (unit_b : b = 1) : 
  ∃ (cut : ℝ → ℝ → Prop), (∀ x y, cut x y → (∃ q r : ℝ, q > 0 ∧ r > 0 ∧ q * r > 1)) :=
sorry

end cube_cut_possible_l190_190966


namespace ducks_cows_legs_l190_190068

theorem ducks_cows_legs (D C : ℕ) (L H X : ℤ)
  (hC : C = 13)
  (hL : L = 2 * D + 4 * C)
  (hH : H = D + C)
  (hCond : L = 3 * H + X) : X = 13 := by
  sorry

end ducks_cows_legs_l190_190068


namespace converse_equivalence_l190_190847

-- Definition of the original proposition
def original_proposition : Prop := ∀ (x : ℝ), x < 0 → x^2 > 0

-- Definition of the converse proposition
def converse_proposition : Prop := ∀ (x : ℝ), x^2 > 0 → x < 0

-- Theorem statement asserting the equivalence
theorem converse_equivalence : (converse_proposition = ¬ original_proposition) :=
sorry

end converse_equivalence_l190_190847


namespace asymptote_of_hyperbola_l190_190029

theorem asymptote_of_hyperbola : 
  ∀ x y : ℝ, (y^2 / 4 - x^2 = 1) → (y = 2 * x) ∨ (y = -2 * x) := 
by
  sorry

end asymptote_of_hyperbola_l190_190029


namespace count_valid_orderings_l190_190886

-- Define the houses and conditions
inductive HouseColor where
  | Green
  | Purple
  | Blue
  | Pink
  | X -- Representing the fifth unspecified house

open HouseColor

def validOrderings : List (List HouseColor) :=
  [
    [Green, Blue, Purple, Pink, X], 
    [Green, Blue, X, Purple, Pink],
    [Green, X, Purple, Blue, Pink],
    [X, Pink, Purple, Blue, Green],
    [X, Purple, Pink, Blue, Green],
    [X, Pink, Blue, Purple, Green]
  ] 

-- Prove that there are exactly 6 valid orderings
theorem count_valid_orderings : (validOrderings.length = 6) :=
by
  -- Since we list all possible valid orderings above, just compute the length
  sorry

end count_valid_orderings_l190_190886


namespace find_a_squared_plus_b_squared_l190_190667

theorem find_a_squared_plus_b_squared (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) : a^2 + b^2 = 7 := 
by
  sorry

end find_a_squared_plus_b_squared_l190_190667


namespace find_x_l190_190785

theorem find_x (x : ℝ) (h : 1 / 7 + 7 / x = 15 / x + 1 / 15) : x = 105 := 
by 
  sorry

end find_x_l190_190785


namespace large_pretzel_cost_l190_190156

theorem large_pretzel_cost : 
  ∀ (P S : ℕ), 
  P = 3 * S ∧ 7 * P + 4 * S = 4 * P + 7 * S + 12 → 
  P = 6 :=
by sorry

end large_pretzel_cost_l190_190156


namespace yangyang_helps_mom_for_5_days_l190_190716

-- Defining the conditions
def quantity_of_rice_in_warehouses_are_same : Prop := sorry
def dad_transports_all_rice_in : ℕ := 10
def mom_transports_all_rice_in : ℕ := 12
def yangyang_transports_all_rice_in : ℕ := 15
def dad_and_mom_start_at_same_time : Prop := sorry
def yangyang_helps_mom_then_dad : Prop := sorry
def finish_transporting_at_same_time : Prop := sorry

-- The theorem to prove
theorem yangyang_helps_mom_for_5_days (h1 : quantity_of_rice_in_warehouses_are_same) 
    (h2 : dad_and_mom_start_at_same_time) 
    (h3 : yangyang_helps_mom_then_dad) 
    (h4 : finish_transporting_at_same_time) : 
    yangyang_helps_mom_then_dad :=
sorry

end yangyang_helps_mom_for_5_days_l190_190716


namespace sqrt_meaningful_range_l190_190786

theorem sqrt_meaningful_range (x : ℝ) : 
  (x + 4) ≥ 0 ↔ x ≥ -4 :=
by sorry

end sqrt_meaningful_range_l190_190786


namespace find_stream_speed_l190_190934

variable (D : ℝ) (v : ℝ)

theorem find_stream_speed 
  (h1 : ∀D v, D / (63 - v) = 2 * (D / (63 + v)))
  (h2 : v = 21) :
  true := 
  by
  sorry

end find_stream_speed_l190_190934


namespace area_of_octagon_in_square_l190_190349

theorem area_of_octagon_in_square : 
  let A := (0, 0)
  let B := (6, 0)
  let C := (6, 6)
  let D := (0, 6)
  let E := (3, 0)
  let F := (6, 3)
  let G := (3, 6)
  let H := (0, 3)
  ∃ (octagon_area : ℚ),
    octagon_area = 6 :=
by
  sorry

end area_of_octagon_in_square_l190_190349


namespace jake_eats_papayas_in_one_week_l190_190710

variable (J : ℕ)
variable (brother_eats : ℕ := 5)
variable (father_eats : ℕ := 4)
variable (total_papayas_in_4_weeks : ℕ := 48)

theorem jake_eats_papayas_in_one_week (h : 4 * (J + brother_eats + father_eats) = total_papayas_in_4_weeks) : J = 3 :=
by
  sorry

end jake_eats_papayas_in_one_week_l190_190710


namespace rhombus_area_eq_54_l190_190787

theorem rhombus_area_eq_54
  (a b : ℝ) (eq_long_side : a = 4 * Real.sqrt 3) (eq_short_side : b = 3 * Real.sqrt 3)
  (rhombus_diagonal1 : ℝ := 9 * Real.sqrt 3) (rhombus_diagonal2 : ℝ := 4 * Real.sqrt 3) :
  (1 / 2) * rhombus_diagonal1 * rhombus_diagonal2 = 54 := by
  sorry

end rhombus_area_eq_54_l190_190787


namespace tom_seashells_found_l190_190790

/-- 
Given:
- sally_seashells = 9 (number of seashells Sally found)
- jessica_seashells = 5 (number of seashells Jessica found)
- total_seashells = 21 (number of seashells found together)

Prove that the number of seashells that Tom found (tom_seashells) is 7.
-/
theorem tom_seashells_found (sally_seashells jessica_seashells total_seashells tom_seashells : ℕ)
  (h₁ : sally_seashells = 9) (h₂ : jessica_seashells = 5) (h₃ : total_seashells = 21) :
  tom_seashells = 7 :=
by
  sorry

end tom_seashells_found_l190_190790


namespace equal_values_on_plane_l190_190992

theorem equal_values_on_plane (f : ℤ × ℤ → ℕ)
    (h_avg : ∀ (i j : ℤ), f (i, j) = (f (i+1, j) + f (i-1, j) + f (i, j+1) + f (i, j-1)) / 4) :
  ∃ c : ℕ, ∀ (i j : ℤ), f (i, j) = c :=
by
  sorry

end equal_values_on_plane_l190_190992


namespace question1_question2_l190_190528

variable (a : ℤ)
def point_P : (ℤ × ℤ) := (2*a - 2, a + 5)

-- Part 1: If point P lies on the x-axis, its coordinates are (-12, 0).
theorem question1 (h1 : a + 5 = 0) : point_P a = (-12, 0) :=
sorry

-- Part 2: If point P lies in the second quadrant and the distance from point P to the x-axis is equal to the distance from point P to the y-axis,
-- the value of a^2023 + 2023 is 2022.
theorem question2 (h2 : 2*a - 2 < 0) (h3 : -(2*a - 2) = a + 5) : a ^ 2023 + 2023 = 2022 :=
sorry

end question1_question2_l190_190528


namespace sequence_expression_l190_190175

theorem sequence_expression (n : ℕ) (h : n ≥ 2) (T : ℕ → ℕ) (a : ℕ → ℕ)
  (hT : ∀ k : ℕ, T k = 2 * k^2)
  (ha : ∀ k : ℕ, k ≥ 2 → a k = T k / T (k - 1)) :
  a n = (n / (n - 1))^2 := 
sorry

end sequence_expression_l190_190175


namespace find_a_l190_190044

theorem find_a (a : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h1 : ∀ x, f (g x) = x)
  (h2 : f x = (Real.log (x + 1) / Real.log 2) + a)
  (h3 : g 4 = 1) :
  a = 3 :=
sorry

end find_a_l190_190044


namespace maximize_profit_l190_190610

noncomputable def profit (x : ℕ) : ℝ :=
  let price := (180 + 10 * x : ℝ)
  let rooms_occupied := (50 - x : ℝ)
  let expenses := 20
  (price - expenses) * rooms_occupied

theorem maximize_profit :
  ∃ x : ℕ, profit x = profit 17 → (180 + 10 * x) = 350 :=
by
  use 17
  sorry

end maximize_profit_l190_190610


namespace least_five_digit_congruent_to_7_mod_17_l190_190209

theorem least_five_digit_congruent_to_7_mod_17 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 17 = 7 ∧ (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 7 → n ≤ m) :=
sorry

end least_five_digit_congruent_to_7_mod_17_l190_190209


namespace original_number_is_142857_l190_190182

-- Definitions based on conditions
def six_digit_number (x : ℕ) : ℕ := 100000 + x
def moved_digit_number (x : ℕ) : ℕ := 10 * x + 1

-- Lean statement of the equivalent problem
theorem original_number_is_142857 : ∃ x, six_digit_number x = 142857 ∧ moved_digit_number x = 3 * six_digit_number x :=
  sorry

end original_number_is_142857_l190_190182


namespace total_handshakes_l190_190238

variable (n : ℕ) (h : n = 12)

theorem total_handshakes (H : ∀ (b : ℕ), b = n → (n * (n - 1)) / 2 = 66) : 
  (12 * 11) / 2 = 66 := 
by
  sorry

end total_handshakes_l190_190238


namespace sequence_general_formula_l190_190819

theorem sequence_general_formula (a : ℕ → ℕ)
    (h1 : a 1 = 3) 
    (h2 : a 2 = 4) 
    (h3 : a 3 = 6) 
    (h4 : a 4 = 10) 
    (h5 : a 5 = 18) :
    ∀ n : ℕ, a n = 2^(n-1) + 2 :=
sorry

end sequence_general_formula_l190_190819


namespace combined_salaries_l190_190493

variable {A B C E : ℝ}
variable (D : ℝ := 7000)
variable (average_salary : ℝ := 8400)
variable (n : ℕ := 5)

theorem combined_salaries (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ E) 
  (h4 : B ≠ C) (h5 : B ≠ E) (h6 : C ≠ E)
  (h7 : average_salary = (A + B + C + D + E) / n) :
  A + B + C + E = 35000 :=
by
  sorry

end combined_salaries_l190_190493


namespace rectangle_lengths_correct_l190_190323

-- Definitions of the parameters and their relationships
noncomputable def AB := 1200
noncomputable def BC := 150
noncomputable def AB_ext := AB
noncomputable def BC_ext := BC + 350
noncomputable def CD := AB
noncomputable def DA := BC

-- Definitions of the calculated distances using the conditions
noncomputable def AP := Real.sqrt (AB^2 + BC_ext^2)
noncomputable def PD := Real.sqrt (BC_ext^2 + AB^2)

-- Using similarity of triangles for PQ and CQ
noncomputable def PQ := (350 / 500) * AP
noncomputable def CQ := (350 / 500) * AB

-- The theorem to prove the final results
theorem rectangle_lengths_correct :
    AP = 1300 ∧
    PD = 1250 ∧
    PQ = 910 ∧
    CQ = 840 :=
    by
    sorry

end rectangle_lengths_correct_l190_190323


namespace p_range_l190_190215

def h (x : ℝ) : ℝ := 2 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem p_range :
  ∀ x, -1 ≤ x ∧ x ≤ 3 → 29 ≤ p x ∧ p x ≤ 93 :=
by
  intros x hx
  sorry

end p_range_l190_190215


namespace no_real_roots_of_quadratic_l190_190199

theorem no_real_roots_of_quadratic (k : ℝ) (hk : k ≠ 0) : ¬ ∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 = 0 :=
by
  sorry

end no_real_roots_of_quadratic_l190_190199


namespace fill_time_with_conditions_l190_190643

-- Define rates as constants
def pipeA_rate := 1 / 10
def pipeB_rate := 1 / 6
def pipeC_rate := 1 / 5
def tarp_factor := 1 / 2
def leak_rate := 1 / 15

-- Define effective fill rate taking into account the tarp and leak
def effective_fill_rate := ((pipeA_rate + pipeB_rate + pipeC_rate) * tarp_factor) - leak_rate

-- Define the required time to fill the pool
def required_time := 1 / effective_fill_rate

theorem fill_time_with_conditions :
  required_time = 6 :=
by
  sorry

end fill_time_with_conditions_l190_190643


namespace find_constants_to_satisfy_equation_l190_190176

-- Define the condition
def equation_condition (x : ℝ) (A B C : ℝ) :=
  -2 * x^2 + 5 * x - 6 = A * (x^2 + 1) + (B * x + C) * x

-- Define the proof problem as a Lean 4 statement
theorem find_constants_to_satisfy_equation (A B C : ℝ) :
  A = -6 ∧ B = 4 ∧ C = 5 ↔ ∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 → equation_condition x A B C := 
by
  sorry

end find_constants_to_satisfy_equation_l190_190176


namespace perimeter_non_shaded_region_l190_190231

def shaded_area : ℤ := 78
def large_rect_area : ℤ := 80
def small_rect_area : ℤ := 8
def total_area : ℤ := large_rect_area + small_rect_area
def non_shaded_area : ℤ := total_area - shaded_area
def non_shaded_width : ℤ := 2
def non_shaded_length : ℤ := non_shaded_area / non_shaded_width
def non_shaded_perimeter : ℤ := 2 * (non_shaded_length + non_shaded_width)

theorem perimeter_non_shaded_region : non_shaded_perimeter = 14 := 
by
  exact rfl

end perimeter_non_shaded_region_l190_190231


namespace find_x_l190_190899

theorem find_x
  (x : ℝ)
  (h : (x + 1) / (x + 5) = (x + 5) / (x + 13)) :
  x = 3 :=
sorry

end find_x_l190_190899


namespace find_x0_l190_190270

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 2 then x^2 - 4 else 2 * x

theorem find_x0 (x0 : ℝ) (h : f x0 = 8) : x0 = 4 := by
  sorry

end find_x0_l190_190270


namespace whisky_replacement_l190_190341

variable (V x : ℝ)

/-- The initial whisky in the jar contains 40% alcohol -/
def initial_volume_of_alcohol (V : ℝ) : ℝ := 0.4 * V

/-- A part (x liters) of this whisky is replaced by another containing 19% alcohol -/
def volume_replaced_whisky (x : ℝ) : ℝ := x
def remaining_whisky (V x : ℝ) : ℝ := V - x

/-- The percentage of alcohol in the jar after replacement is 24% -/
def final_volume_of_alcohol (V x : ℝ) : ℝ := 0.4 * (remaining_whisky V x) + 0.19 * (volume_replaced_whisky x)

/- Prove that the quantity of whisky replaced is 0.16/0.21 times the total volume -/
theorem whisky_replacement :
  final_volume_of_alcohol V x = 0.24 * V → x = (0.16 / 0.21) * V :=
by sorry

end whisky_replacement_l190_190341


namespace friends_lunch_spending_l190_190909

-- Problem conditions and statement to prove
theorem friends_lunch_spending (x : ℝ) (h1 : x + (x + 15) + (x - 20) + 2 * x = 100) : 
  x = 21 :=
by sorry

end friends_lunch_spending_l190_190909


namespace gcd_360_150_l190_190490

theorem gcd_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end gcd_360_150_l190_190490


namespace find_x_l190_190870

variable (x : ℤ)

-- Define the conditions based on the problem
def adjacent_sum_condition := 
  (x + 15) + (x + 8) + (x - 7) = x

-- State the goal, which is to prove x = -8
theorem find_x : x = -8 :=
by
  have h : adjacent_sum_condition x := sorry
  sorry

end find_x_l190_190870


namespace problem_inequality_A_problem_inequality_B_problem_inequality_D_problem_inequality_E_l190_190740

variable {a b c : ℝ}

theorem problem_inequality_A (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a * b < b * c :=
by sorry

theorem problem_inequality_B (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a * c < b * c :=
by sorry

theorem problem_inequality_D (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a + b < b + c :=
by sorry

theorem problem_inequality_E (h1 : a > 0) (h2 : a < b) (h3 : b < c) : c / a > 1 :=
by sorry

end problem_inequality_A_problem_inequality_B_problem_inequality_D_problem_inequality_E_l190_190740


namespace hcf_of_two_numbers_is_18_l190_190033

theorem hcf_of_two_numbers_is_18
  (product : ℕ)
  (lcm : ℕ)
  (hcf : ℕ) :
  product = 571536 ∧ lcm = 31096 → hcf = 18 := 
by sorry

end hcf_of_two_numbers_is_18_l190_190033


namespace batsman_average_after_17th_l190_190861

def runs_17th_inning : ℕ := 87
def increase_in_avg : ℕ := 4
def num_innings : ℕ := 17

theorem batsman_average_after_17th (A : ℕ) (H : A + increase_in_avg = (16 * A + runs_17th_inning) / num_innings) : 
  (A + increase_in_avg) = 23 := sorry

end batsman_average_after_17th_l190_190861


namespace find_k_value_l190_190241

variables {R : Type*} [Field R] {a b x k : R}

-- Definitions for the conditions in the problem
def f (x : R) (a b : R) : R := (b * x + 1) / (2 * x + a)

-- Statement of the problem
theorem find_k_value (h_ab : a * b ≠ 2)
  (h_k : ∀ (x : R), x ≠ 0 → f x a b * f (x⁻¹) a b = k) :
  k = (1 : R) / 4 :=
by
  sorry

end find_k_value_l190_190241


namespace fraction_meaningful_condition_l190_190360

theorem fraction_meaningful_condition (x : ℝ) : (∃ y, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end fraction_meaningful_condition_l190_190360


namespace total_weekly_earnings_l190_190850

-- Define the total weekly hours and earnings
def weekly_hours_weekday : ℕ := 5 * 5
def weekday_rate : ℕ := 3
def weekday_earnings : ℕ := weekly_hours_weekday * weekday_rate

-- Define the total weekend hours and earnings
def weekend_days : ℕ := 2
def weekend_hours_per_day : ℕ := 3
def weekend_rate : ℕ := 3 * 2
def weekend_hours : ℕ := weekend_days * weekend_hours_per_day
def weekend_earnings : ℕ := weekend_hours * weekend_rate

-- Prove that Mitch's total earnings per week are $111
theorem total_weekly_earnings : weekday_earnings + weekend_earnings = 111 := by
  sorry

end total_weekly_earnings_l190_190850


namespace ladder_base_distance_l190_190591

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l190_190591


namespace comparison_of_exponential_values_l190_190227

theorem comparison_of_exponential_values : 
  let a := 0.3^3
  let b := 3^0.3
  let c := 0.2^3
  c < a ∧ a < b := 
by 
  let a := 0.3^3
  let b := 3^0.3
  let c := 0.2^3
  sorry

end comparison_of_exponential_values_l190_190227


namespace f_is_constant_l190_190738

noncomputable def is_const (f : ℤ × ℤ → ℕ) := ∃ c : ℕ, ∀ p : ℤ × ℤ, f p = c

theorem f_is_constant (f : ℤ × ℤ → ℕ) 
  (h : ∀ x y : ℤ, 4 * f (x, y) = f (x - 1, y) + f (x, y + 1) + f (x + 1, y) + f (x, y - 1)) :
  is_const f :=
sorry

end f_is_constant_l190_190738


namespace exists_n_for_binomial_congruence_l190_190040

theorem exists_n_for_binomial_congruence 
  (p : ℕ) (a k : ℕ) (prime_p : Nat.Prime p) 
  (positive_a : a > 0) (positive_k : k > 0)
  (h1 : p^a < k) (h2 : k < 2 * p^a) : 
  ∃ n : ℕ, n < p^(2 * a) ∧ (Nat.choose n k) % p^a = n % p^a ∧ n % p^a = k % p^a :=
by
  sorry

end exists_n_for_binomial_congruence_l190_190040


namespace denominator_exceeds_numerator_by_263_l190_190523

def G : ℚ := 736 / 999

theorem denominator_exceeds_numerator_by_263 : 999 - 736 = 263 := by
  -- Since 736 / 999 is the simplest form already, we simply state the obvious difference
  rfl

end denominator_exceeds_numerator_by_263_l190_190523


namespace yerema_can_pay_exactly_l190_190119

theorem yerema_can_pay_exactly (t k b m : ℤ) 
    (h_foma : 3 * t + 4 * k + 5 * b = 11 * m) : 
    ∃ n : ℤ, 9 * t + k + 4 * b = 11 * n := 
by 
    sorry

end yerema_can_pay_exactly_l190_190119


namespace problem1_problem2_l190_190777

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a/x

/-- 
Given the function f(x) = ln(x) + a/x (where a is a real number),
prove that if the function f(x) has two zeros, then 0 < a < 1/e.
-/
theorem problem1 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → (0 < a ∧ a < 1/Real.exp 1) :=
sorry

/-- 
Given the function f(x) = ln(x) + a/x (where a is a real number) and a line y = m
that intersects the graph of f(x) at two points (x1, m) and (x2, m),
prove that x1 + x2 > 2a.
-/
theorem problem2 (x1 x2 a m : ℝ) (h : f x1 a = m ∧ f x2 a = m ∧ x1 ≠ x2) :
  x1 + x2 > 2 * a :=
sorry

end problem1_problem2_l190_190777


namespace number_of_rabbits_l190_190735

theorem number_of_rabbits (x y : ℕ) (h1 : x + y = 28) (h2 : 4 * x = 6 * y + 12) : x = 18 :=
by
  sorry

end number_of_rabbits_l190_190735


namespace smallest_n_contains_constant_term_l190_190324

theorem smallest_n_contains_constant_term :
  ∃ n : ℕ, (∀ x : ℝ, x ≠ 0 → (2 * x^3 + 1 / x^(1/2))^n = c ↔ n = 7) :=
by
  sorry

end smallest_n_contains_constant_term_l190_190324


namespace correct_calculation_l190_190719

theorem correct_calculation (a b : ℝ) : 
  ¬(3 * a + b = 3 * a * b) ∧ 
  ¬(a^2 + a^2 = a^4) ∧ 
  ¬((a - b)^2 = a^2 - b^2) ∧ 
  ((-3 * a)^2 = 9 * a^2) :=
by
  sorry

end correct_calculation_l190_190719


namespace correct_option_l190_190247

theorem correct_option : 
  (-(2:ℤ))^3 ≠ -6 ∧ 
  (-(1:ℤ))^10 ≠ -10 ∧ 
  (-(1:ℚ)/3)^3 ≠ -1/9 ∧ 
  -(2:ℤ)^2 = -4 :=
by 
  sorry

end correct_option_l190_190247


namespace problem_1_max_value_problem_2_good_sets_count_l190_190098

noncomputable def goodSetMaxValue : ℤ :=
  2012

noncomputable def goodSetCount : ℤ :=
  1006

theorem problem_1_max_value {M : Set ℤ} (hM : ∀ x, x ∈ M ↔ |x| ≤ 2014) :
  ∀ a b c : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (1 / a + 1 / b = 2 / c) →
  (a + c = 2 * b) →
  a ∈ M ∧ b ∈ M ∧ c ∈ M →
  ∃ P : Set ℤ, P = {a, b, c} ∧ a ∈ P ∧ b ∈ P ∧ c ∈ P ∧
  goodSetMaxValue = 2012 :=
sorry

theorem problem_2_good_sets_count {M : Set ℤ} (hM : ∀ x, x ∈ M ↔ |x| ≤ 2014) :
  ∀ a b c : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (1 / a + 1 / b = 2 / c) →
  (a + c = 2 * b) →
  a ∈ M ∧ b ∈ M ∧ c ∈ M →
  ∃ P : Set ℤ, P = {a, b, c} ∧ a ∈ P ∧ b ∈ P ∧ c ∈ P ∧
  goodSetCount = 1006 :=
sorry

end problem_1_max_value_problem_2_good_sets_count_l190_190098


namespace hyperbola_sufficient_but_not_necessary_asymptote_l190_190510

-- Define the equation of the hyperbola and the related asymptotes
def hyperbola_eq (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def asymptote_eq (a b x y : ℝ) : Prop :=
  y = b / a * x ∨ y = - (b / a * x)

-- Stating the theorem that expresses the sufficiency but not necessity
theorem hyperbola_sufficient_but_not_necessary_asymptote (a b : ℝ) :
  (∃ x y, hyperbola_eq a b x y) → (∀ x y, asymptote_eq a b x y) ∧ ¬ (∀ x y, (asymptote_eq a b x y) → (hyperbola_eq a b x y)) := 
sorry

end hyperbola_sufficient_but_not_necessary_asymptote_l190_190510


namespace find_whole_number_M_l190_190827

-- Define the conditions
def condition (M : ℕ) : Prop :=
  21 < M ∧ M < 23

-- Define the main theorem to be proven
theorem find_whole_number_M (M : ℕ) (h : condition M) : M = 22 := by
  sorry

end find_whole_number_M_l190_190827


namespace solve_fractional_equation_l190_190860

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 2) : 
  (4 * x ^ 2 + 3 * x + 2) / (x - 2) = 4 * x + 5 ↔ x = -2 := by 
  sorry

end solve_fractional_equation_l190_190860


namespace promoting_cashback_beneficial_for_bank_cashback_in_rubles_preferable_l190_190562

-- Definitions of conditions
def attracts_new_clients (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (new_clients : Prop), new_clients

def promotes_partnerships (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (partnerships : Prop), partnerships

def enhances_competitiveness (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (competitiveness : Prop), competitiveness

def liquidity_advantage (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (liquidity : Prop), liquidity

def no_expiry_concerns (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (no_expiry : Prop), no_expiry

def no_partner_limitations (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (partner_limitations : Prop), ¬partner_limitations

-- Lean statements for the proof problems
theorem promoting_cashback_beneficial_for_bank (bank_promotes_cashback : Prop) :
  attracts_new_clients bank_promotes_cashback ∧
  promotes_partnerships bank_promotes_cashback ∧ 
  enhances_competitiveness bank_promotes_cashback →
  bank_promotes_cashback := 
sorry

theorem cashback_in_rubles_preferable (cashback_rubles : Prop) :
  liquidity_advantage cashback_rubles ∧
  no_expiry_concerns cashback_rubles ∧
  no_partner_limitations cashback_rubles →
  cashback_rubles :=
sorry

end promoting_cashback_beneficial_for_bank_cashback_in_rubles_preferable_l190_190562


namespace no_positive_real_roots_l190_190150

theorem no_positive_real_roots (x : ℝ) : (x^3 + 6 * x^2 + 11 * x + 6 = 0) → x < 0 :=
sorry

end no_positive_real_roots_l190_190150


namespace initial_wine_volume_l190_190180

theorem initial_wine_volume (x : ℝ) 
  (h₁ : ∀ k : ℝ, k = x → ∀ n : ℕ, n = 3 → 
    (∀ y : ℝ, y = k - 4 * (1 - ((k - 4) / k) ^ n) + 2.5)) :
  x = 16 := by
  sorry

end initial_wine_volume_l190_190180


namespace finalStoresAtEndOf2020_l190_190432

def initialStores : ℕ := 23
def storesOpened2019 : ℕ := 5
def storesClosed2019 : ℕ := 2
def storesOpened2020 : ℕ := 10
def storesClosed2020 : ℕ := 6

theorem finalStoresAtEndOf2020 : initialStores + (storesOpened2019 - storesClosed2019) + (storesOpened2020 - storesClosed2020) = 30 :=
by
  sorry

end finalStoresAtEndOf2020_l190_190432


namespace inequality_not_true_l190_190540

theorem inequality_not_true (a b : ℝ) (h1 : a < b) (h2 : b < 0) : ¬ (a > 0) :=
sorry

end inequality_not_true_l190_190540


namespace problem1_problem2_l190_190970

-- Definitions of the three conditions given
def condition1 (x y : Nat) : Prop := x > y
def condition2 (y z : Nat) : Prop := y > z
def condition3 (x z : Nat) : Prop := 2 * z > x

-- Problem 1: If the number of teachers is 4, prove the maximum number of female students is 6.
theorem problem1 (z : Nat) (hz : z = 4) : ∃ y : Nat, (∀ x : Nat, condition1 x y → condition2 y z → condition3 x z) ∧ y = 6 :=
by
  sorry

-- Problem 2: Prove the minimum number of people in the group is 12.
theorem problem2 : ∃ z x y : Nat, (condition1 x y ∧ condition2 y z ∧ condition3 x z ∧ z < y ∧ y < x ∧ x < 2 * z) ∧ z = 3 ∧ x = 5 ∧ y = 4 ∧ x + y + z = 12 :=
by
  sorry

end problem1_problem2_l190_190970


namespace factor_2210_two_digit_l190_190229

theorem factor_2210_two_digit :
  (∃ (a b : ℕ), a * b = 2210 ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99) ∧
  (∃ (c d : ℕ), c * d = 2210 ∧ 10 ≤ c ∧ c ≤ 99 ∧ 10 ≤ d ∧ d ≤ 99) ∧
  (∀ (x y : ℕ), x * y = 2210 ∧ 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 → 
   ((x = c ∧ y = d) ∨ (x = d ∧ y = c) ∨ (x = a ∧ y = b) ∨ (x = b ∧ y = a))) :=
sorry

end factor_2210_two_digit_l190_190229


namespace average_age_of_9_students_l190_190124

theorem average_age_of_9_students (avg_age_17_students : ℕ)
                                   (num_students : ℕ)
                                   (avg_age_5_students : ℕ)
                                   (num_5_students : ℕ)
                                   (age_17th_student : ℕ) :
    avg_age_17_students = 17 →
    num_students = 17 →
    avg_age_5_students = 14 →
    num_5_students = 5 →
    age_17th_student = 75 →
    (144 / 9) = 16 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_age_of_9_students_l190_190124


namespace angle_APB_l190_190132

-- Define the problem conditions
variables (XY : Π X Y : ℝ, XY = X + Y) -- Line XY is a straight line
          (semicircle_XAZ : Π X A Z : ℝ, semicircle_XAZ = X + Z - A) -- Semicircle XAZ
          (semicircle_ZBY : Π Z B Y : ℝ, semicircle_ZBY = Z + Y - B) -- Semicircle ZBY
          (PA_tangent_XAZ_at_A : Π P A X Z : ℝ, PA_tangent_XAZ_at_A = P + A + X - Z) -- PA tangent to XAZ at A
          (PB_tangent_ZBY_at_B : Π P B Z Y : ℝ, PB_tangent_ZBY_at_B = P + B + Z - Y) -- PB tangent to ZBY at B
          (arc_XA : ℝ := 45) -- Arc XA is 45 degrees
          (arc_BY : ℝ := 60) -- Arc BY is 60 degrees

-- Main theorem to prove
theorem angle_APB : ∀ P A B: ℝ, 
  540 - 90 - 135 - 120 - 90 = 105 := by 
  -- Proof goes here
  sorry

end angle_APB_l190_190132


namespace encore_songs_l190_190633

-- Definitions corresponding to the conditions
def repertoire_size : ℕ := 30
def first_set_songs : ℕ := 5
def second_set_songs : ℕ := 7
def average_songs_per_set_3_and_4 : ℕ := 8

-- The statement to prove
theorem encore_songs : (repertoire_size - (first_set_songs + second_set_songs)) - (2 * average_songs_per_set_3_and_4) = 2 := by
  sorry

end encore_songs_l190_190633


namespace triangle_smallest_side_l190_190842

theorem triangle_smallest_side (a b c : ℝ) (h : b^2 + c^2 ≥ 5 * a^2) : 
    (a ≤ b ∧ a ≤ c) := 
sorry

end triangle_smallest_side_l190_190842


namespace excircle_identity_l190_190406

variables (a b c r_a r_b r_c : ℝ)

-- Conditions: r_a, r_b, r_c are the radii of the excircles opposite vertices A, B, and C respectively.
-- In the triangle ABC, a, b, c are the sides opposite vertices A, B, and C respectively.

theorem excircle_identity:
  (a^2 / (r_a * (r_b + r_c)) + b^2 / (r_b * (r_c + r_a)) + c^2 / (r_c * (r_a + r_b))) = 2 :=
by
  sorry

end excircle_identity_l190_190406


namespace count_4x4_increasing_arrays_l190_190652

-- Define the notion of a 4x4 grid that satisfies the given conditions
def isInIncreasingOrder (matrix : (Fin 4) → (Fin 4) → Nat) : Prop :=
  (∀ i j : Fin 4, i < 3 -> matrix i j < matrix (i+1) j) ∧
  (∀ i j : Fin 4, j < 3 -> matrix i j < matrix i (j+1))

def validGrid (matrix : (Fin 4) → (Fin 4) → Nat) : Prop :=
  (∀ i j : Fin 4, 1 ≤ matrix i j ∧ matrix i j ≤ 16) ∧ isInIncreasingOrder matrix

noncomputable def countValidGrids : ℕ :=
  sorry

theorem count_4x4_increasing_arrays : countValidGrids = 13824 :=
  sorry

end count_4x4_increasing_arrays_l190_190652


namespace customers_who_didnt_tip_l190_190183

theorem customers_who_didnt_tip:
  ∀ (total_customers tips_per_customer total_tips : ℕ),
  total_customers = 10 →
  tips_per_customer = 3 →
  total_tips = 15 →
  (total_customers - total_tips / tips_per_customer) = 5 :=
by
  intros
  sorry

end customers_who_didnt_tip_l190_190183


namespace ninth_square_more_than_eighth_l190_190595

noncomputable def side_length (n : ℕ) : ℕ := 3 + 2 * (n - 1)

noncomputable def tile_count (n : ℕ) : ℕ := (side_length n) ^ 2

theorem ninth_square_more_than_eighth : (tile_count 9 - tile_count 8) = 72 :=
by sorry

end ninth_square_more_than_eighth_l190_190595


namespace not_possible_for_runners_in_front_l190_190969

noncomputable def runnerInFrontAtAnyMoment 
  (track_length : ℝ)
  (stands_length : ℝ)
  (runners_speeds : Fin 10 → ℝ) : Prop := 
  ∀ t : ℝ, ∃ i : Fin 10, 
  ∃ n : ℤ, 
  (runners_speeds i * t - n * track_length) % track_length ≤ stands_length

theorem not_possible_for_runners_in_front 
  (track_length stands_length : ℝ)
  (runners_speeds : Fin 10 → ℝ) 
  (h_track : track_length = 1)
  (h_stands : stands_length = 0.1)
  (h_speeds : ∀ i : Fin 10, 20 + i = runners_speeds i) : 
  ¬ runnerInFrontAtAnyMoment track_length stands_length runners_speeds :=
sorry

end not_possible_for_runners_in_front_l190_190969


namespace length_of_one_side_of_square_l190_190904

variable (total_ribbon_length : ℕ) (triangle_perimeter : ℕ)

theorem length_of_one_side_of_square (h1 : total_ribbon_length = 78)
                                    (h2 : triangle_perimeter = 46) :
  (total_ribbon_length - triangle_perimeter) / 4 = 8 :=
by
  sorry

end length_of_one_side_of_square_l190_190904


namespace vertex_of_parabola_l190_190676

theorem vertex_of_parabola 
  (a b c : ℝ) 
  (h1 : a * 2^2 + b * 2 + c = 5)
  (h2 : -b / (2 * a) = 2) : 
  (2, 4 * a + 2 * b + c) = (2, 5) :=
by
  sorry

end vertex_of_parabola_l190_190676


namespace lowest_number_in_range_l190_190840

theorem lowest_number_in_range (y : ℕ) (h : ∀ x y : ℕ, 0 < x ∧ x < y) : ∃ x : ℕ, x = 999 :=
by
  existsi 999
  sorry

end lowest_number_in_range_l190_190840


namespace simplify_polynomial_l190_190066

variable (x : ℝ)

theorem simplify_polynomial :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 6*x^3 =
  6*x^3 - x^2 + 23*x - 3 :=
by
  sorry

end simplify_polynomial_l190_190066


namespace lottery_most_frequent_number_l190_190281

noncomputable def m (i : ℕ) : ℚ :=
  ((i - 1) * (90 - i) * (89 - i) * (88 - i)) / 6

theorem lottery_most_frequent_number :
  ∀ (i : ℕ), 2 ≤ i ∧ i ≤ 87 → m 23 ≥ m i :=
by 
  sorry -- Proof goes here. This placeholder allows the file to compile.

end lottery_most_frequent_number_l190_190281


namespace peanuts_in_box_after_addition_l190_190491

theorem peanuts_in_box_after_addition : 4 + 12 = 16 := by
  sorry

end peanuts_in_box_after_addition_l190_190491


namespace company_employees_count_l190_190332

theorem company_employees_count :
  (females : ℕ) ->
  (advanced_degrees : ℕ) ->
  (college_degree_only_males : ℕ) ->
  (advanced_degrees_females : ℕ) ->
  (110 = females) ->
  (90 = advanced_degrees) ->
  (35 = college_degree_only_males) ->
  (55 = advanced_degrees_females) ->
  (females - advanced_degrees_females + college_degree_only_males + advanced_degrees = 180) :=
by
  intros females advanced_degrees college_degree_only_males advanced_degrees_females
  intro h_females h_advanced_degrees h_college_degree_only_males h_advanced_degrees_females
  sorry

end company_employees_count_l190_190332


namespace ratio_a_to_c_l190_190923

variable (a b c : ℚ)

theorem ratio_a_to_c (h1 : a / b = 7 / 3) (h2 : b / c = 1 / 5) : a / c = 7 / 15 := 
sorry

end ratio_a_to_c_l190_190923


namespace club_members_l190_190018

theorem club_members (M W : ℕ) (h1 : M + W = 30) (h2 : M + 1/3 * (W : ℝ) = 18) : M = 12 :=
by
  -- proof step
  sorry

end club_members_l190_190018


namespace number_of_red_balloons_l190_190675

-- Definitions for conditions
def balloons_total : ℕ := 85
def at_least_one_red (red blue : ℕ) : Prop := red ≥ 1 ∧ red + blue = balloons_total
def every_pair_has_blue (red blue : ℕ) : Prop := ∀ r1 r2, r1 < red → r2 < red → red = 1

-- Theorem to be proved
theorem number_of_red_balloons (red blue : ℕ) 
  (total : red + blue = balloons_total)
  (at_least_one : at_least_one_red red blue)
  (pair_condition : every_pair_has_blue red blue) : red = 1 :=
sorry

end number_of_red_balloons_l190_190675


namespace smallest_x_plus_y_l190_190556

theorem smallest_x_plus_y (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) (h_eq : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 15) : x + y = 64 :=
sorry

end smallest_x_plus_y_l190_190556


namespace reduced_price_per_dozen_is_3_l190_190337

variable (P : ℝ) -- original price of an apple
variable (R : ℝ) -- reduced price of an apple
variable (A : ℝ) -- number of apples originally bought for Rs. 40
variable (cost_per_dozen_reduced : ℝ) -- reduced price per dozen apples

-- Define the conditions
axiom reduction_condition : R = 0.60 * P
axiom apples_bought_condition : 40 = A * P
axiom more_apples_condition : 40 = (A + 64) * R

-- Define the proof problem
theorem reduced_price_per_dozen_is_3 : cost_per_dozen_reduced = 3 :=
by
  sorry

end reduced_price_per_dozen_is_3_l190_190337


namespace circumcenter_coords_l190_190154

-- Define the given points A, B, and C
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (-5, 1)
def C : ℝ × ℝ := (3, -5)

-- The target statement to prove
theorem circumcenter_coords :
  ∃ x y : ℝ, (x - 2)^2 + (y - 2)^2 = (x + 5)^2 + (y - 1)^2 ∧
             (x - 2)^2 + (y - 2)^2 = (x - 3)^2 + (y + 5)^2 ∧
             x = -1 ∧ y = -2 :=
by
  sorry

end circumcenter_coords_l190_190154


namespace largest_element_lg11_l190_190833

variable (x y : ℝ)
variable (A : Set ℝ)  (B : Set ℝ)

-- Conditions
def condition1 : A = Set.insert (Real.log x) (Set.insert (Real.log y) (Set.insert (Real.log (x + y / x)) ∅)) := sorry
def condition2 : B = Set.insert 0 (Set.insert 1 ∅) := sorry
def condition3 : B ⊆ A := sorry

-- Statement
theorem largest_element_lg11 (x y : ℝ)

  (Aeq : A = Set.insert (Real.log x) (Set.insert (Real.log y) (Set.insert (Real.log (x + y / x)) ∅)))
  (Beq : B = Set.insert 0 (Set.insert 1 ∅))
  (subset : B ⊆ A) :
  ∃ M ∈ A, ∀ a ∈ A, a ≤ M ∧ M = Real.log 11 :=
sorry

end largest_element_lg11_l190_190833


namespace percentage_loss_l190_190522

variable (CP SP : ℝ) (Loss : ℝ := CP - SP) (Percentage_of_Loss : ℝ := (Loss / CP) * 100)

theorem percentage_loss (h1: CP = 1600) (h2: SP = 1440) : Percentage_of_Loss = 10 := by
  sorry

end percentage_loss_l190_190522


namespace closest_whole_number_l190_190805

theorem closest_whole_number :
  let x := (10^2001 + 10^2003) / (10^2002 + 10^2002)
  abs ((x : ℝ) - 5) < 1 :=
by 
  sorry

end closest_whole_number_l190_190805


namespace hunter_rats_l190_190286

-- Defining the conditions
variable (H : ℕ) (E : ℕ := H + 30) (K : ℕ := 3 * (H + E)) 
  
-- Defining the total number of rats condition
def total_rats : Prop := H + E + K = 200

-- Defining the goal: Prove Hunter has 10 rats
theorem hunter_rats (h : total_rats H) : H = 10 := by
  sorry

end hunter_rats_l190_190286


namespace find_value_l190_190733

-- Given conditions of the problem
axiom condition : ∀ (a : ℝ), a - 1/a = 1

-- The mathematical proof problem
theorem find_value (a : ℝ) (h : a - 1/a = 1) : a^2 - a + 2 = 3 :=
by
  sorry

end find_value_l190_190733


namespace volume_of_rect_prism_l190_190769

variables {a b c V : ℝ}

theorem volume_of_rect_prism :
  (∃ (a b c : ℝ), (a * b = Real.sqrt 2) ∧ (b * c = Real.sqrt 3) ∧ (a * c = Real.sqrt 6) ∧ V = a * b * c) →
  V = Real.sqrt 6 :=
by
  sorry

end volume_of_rect_prism_l190_190769


namespace min_value_inequality_l190_190642

theorem min_value_inequality (x y : ℝ) (h1 : x^2 + y^2 = 3) (h2 : |x| ≠ |y|) :
  ∃ (m : ℝ), m = (1 / (2*x + y)^2 + 4 / (x - 2*y)^2) ∧ m = 3 / 5 :=
by
  sorry

end min_value_inequality_l190_190642


namespace book_cost_l190_190365

-- Define the problem parameters
variable (p : ℝ) -- cost of one book in dollars

-- Conditions given in the problem
def seven_copies_cost_less_than_15 (p : ℝ) : Prop := 7 * p < 15
def eleven_copies_cost_more_than_22 (p : ℝ) : Prop := 11 * p > 22

-- The theorem stating the cost is between the given bounds
theorem book_cost (p : ℝ) (h1 : seven_copies_cost_less_than_15 p) (h2 : eleven_copies_cost_more_than_22 p) : 
    2 < p ∧ p < (15 / 7 : ℝ) :=
sorry

end book_cost_l190_190365


namespace option_d_is_correct_l190_190764

theorem option_d_is_correct {x y : ℝ} (h : x - 2 = y - 2) : x = y := 
by 
  sorry

end option_d_is_correct_l190_190764


namespace debbys_sister_candy_l190_190348

-- Defining the conditions
def debby_candy : ℕ := 32
def eaten_candy : ℕ := 35
def remaining_candy : ℕ := 39

-- The proof problem
theorem debbys_sister_candy : ∃ S : ℕ, debby_candy + S - eaten_candy = remaining_candy → S = 42 :=
by
  sorry  -- The proof goes here

end debbys_sister_candy_l190_190348


namespace number_of_interviewees_l190_190269

theorem number_of_interviewees (n : ℕ) (h : (6 : ℚ) / (n * (n - 1)) = 1 / 70) : n = 21 :=
sorry

end number_of_interviewees_l190_190269


namespace correct_time_l190_190747

-- Define the observed times on the clocks
def time1 := 14 * 60 + 54  -- 14:54 in minutes
def time2 := 14 * 60 + 57  -- 14:57 in minutes
def time3 := 15 * 60 + 2   -- 15:02 in minutes
def time4 := 15 * 60 + 3   -- 15:03 in minutes

-- Define the inaccuracies of the clocks
def inaccuracy1 := 2  -- First clock off by 2 minutes
def inaccuracy2 := 3  -- Second clock off by 3 minutes
def inaccuracy3 := -4  -- Third clock off by 4 minutes
def inaccuracy4 := -5  -- Fourth clock off by 5 minutes

-- State that given these conditions, the correct time is 14:58
theorem correct_time : ∃ (T : Int), 
  (time1 + inaccuracy1 = T) ∧
  (time2 + inaccuracy2 = T) ∧
  (time3 + inaccuracy3 = T) ∧
  (time4 + inaccuracy4 = T) ∧
  (T = 14 * 60 + 58) :=
by
  sorry

end correct_time_l190_190747


namespace find_quotient_l190_190333

theorem find_quotient (A : ℕ) (h : 41 = (5 * A) + 1) : A = 8 :=
by
  sorry

end find_quotient_l190_190333


namespace price_of_second_candy_l190_190173

variables (X P : ℝ)

-- Conditions
def total_weight (X : ℝ) := X + 6.25 = 10
def total_value (X P : ℝ) := 3.50 * X + 6.25 * P = 40

-- Proof problem
theorem price_of_second_candy (h1 : total_weight X) (h2 : total_value X P) : P = 4.30 :=
by 
  sorry

end price_of_second_candy_l190_190173


namespace gcd_of_ratio_and_lcm_l190_190135

theorem gcd_of_ratio_and_lcm (A B : ℕ) (k : ℕ) (hA : A = 5 * k) (hB : B = 6 * k) (hlcm : Nat.lcm A B = 180) : Nat.gcd A B = 6 :=
by
  sorry

end gcd_of_ratio_and_lcm_l190_190135


namespace limit_example_l190_190683

theorem limit_example (ε : ℝ) (hε : 0 < ε) :
  ∃ δ : ℝ, 0 < δ ∧ 
  (∀ x : ℝ, 0 < |x - 1/2| ∧ |x - 1/2| < δ →
    |((2 * x^2 - 5 * x + 2) / (x - 1/2)) + 3| < ε) :=
sorry -- The proof is not provided

end limit_example_l190_190683


namespace solution_set_of_inequality_l190_190905

theorem solution_set_of_inequality :
  { x : ℝ | abs (x - 4) + abs (3 - x) < 2 } = { x : ℝ | 2.5 < x ∧ x < 4.5 } := sorry

end solution_set_of_inequality_l190_190905


namespace correct_calculation_l190_190197

variable (a b : ℝ)

theorem correct_calculation :
  -(a - b) = -a + b := by
  sorry

end correct_calculation_l190_190197


namespace two_digit_prime_sum_9_l190_190766

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- There are 0 two-digit prime numbers for which the sum of the digits equals 9 -/
theorem two_digit_prime_sum_9 : ∃! n : ℕ, (9 ≤ n ∧ n < 100) ∧ (n.digits 10).sum = 9 ∧ is_prime n :=
sorry

end two_digit_prime_sum_9_l190_190766


namespace isosceles_triangle_length_l190_190009

theorem isosceles_triangle_length (BC : ℕ) (area : ℕ) (h : ℕ)
  (isosceles : AB = AC)
  (BC_val : BC = 16)
  (area_val : area = 120)
  (height_val : h = (2 * area) / BC)
  (AB_square : ∀ BD AD : ℕ, BD = BC / 2 → AD = h → AB^2 = AD^2 + BD^2)
  : AB = 17 :=
by
  sorry

end isosceles_triangle_length_l190_190009


namespace problem_statement_l190_190541

theorem problem_statement (x y : ℕ) (h1 : x = 3) (h2 :y = 5) :
  (x^5 + 2*y^2 - 15) / 7 = 39 + 5 / 7 := 
by 
  sorry

end problem_statement_l190_190541


namespace prove_mouse_cost_l190_190557

variable (M K : ℕ)

theorem prove_mouse_cost (h1 : K = 3 * M) (h2 : M + K = 64) : M = 16 :=
by
  sorry

end prove_mouse_cost_l190_190557


namespace solve_for_y_l190_190506

theorem solve_for_y (y : ℝ) : (10 - y) ^ 2 = 4 * y ^ 2 → y = 10 / 3 ∨ y = -10 :=
by
  intro h
  -- The proof steps would go here, but we include sorry to allow for compilation.
  sorry

end solve_for_y_l190_190506


namespace x_add_inv_ge_two_x_add_inv_eq_two_iff_l190_190410

theorem x_add_inv_ge_two {x : ℝ} (h : 0 < x) : x + (1 / x) ≥ 2 :=
sorry

theorem x_add_inv_eq_two_iff {x : ℝ} (h : 0 < x) : (x + (1 / x) = 2) ↔ (x = 1) :=
sorry

end x_add_inv_ge_two_x_add_inv_eq_two_iff_l190_190410


namespace average_price_of_returned_cans_l190_190604

theorem average_price_of_returned_cans (total_cans : ℕ) (returned_cans : ℕ) (remaining_cans : ℕ)
  (avg_price_total : ℚ) (avg_price_remaining : ℚ) :
  total_cans = 6 →
  returned_cans = 2 →
  remaining_cans = 4 →
  avg_price_total = 36.5 →
  avg_price_remaining = 30 →
  (avg_price_total * total_cans - avg_price_remaining * remaining_cans) / returned_cans = 49.5 :=
by
  intros h_total_cans h_returned_cans h_remaining_cans h_avg_price_total h_avg_price_remaining
  rw [h_total_cans, h_returned_cans, h_remaining_cans, h_avg_price_total, h_avg_price_remaining]
  sorry

end average_price_of_returned_cans_l190_190604


namespace average_weight_b_c_l190_190020

theorem average_weight_b_c (A B C : ℝ) (h1 : A + B + C = 126) (h2 : A + B = 80) (h3 : B = 40) : 
  (B + C) / 2 = 43 := 
by 
  -- Proof would go here, but is left as sorry as per instructions
  sorry

end average_weight_b_c_l190_190020


namespace find_b_l190_190930

noncomputable def given_c := 3
noncomputable def given_C := Real.pi / 3
noncomputable def given_cos_C := 1 / 2
noncomputable def given_a (b : ℝ) := 2 * b

theorem find_b (b : ℝ) (h1 : given_c = 3) (h2 : given_cos_C = Real.cos (given_C)) (h3 : given_a b = 2 * b) : b = Real.sqrt 3 := 
by
  sorry

end find_b_l190_190930


namespace problem_statement_l190_190421

theorem problem_statement
  (a b c d e : ℝ)
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |e| = 1) :
  e^2 + 2023 * (c * d) - (a + b) / 20 = 2024 := 
by 
  sorry

end problem_statement_l190_190421


namespace fraction_to_decimal_l190_190232

theorem fraction_to_decimal :
  (7 : ℝ) / 16 = 0.4375 := 
by sorry

end fraction_to_decimal_l190_190232


namespace trigonometric_identity_l190_190239

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6) ^ 2 = -(2 + Real.sqrt 3) / 3 := 
sorry

end trigonometric_identity_l190_190239


namespace range_of_a_l190_190050

theorem range_of_a (a : ℝ) : 
  (∀ x1 x2 : ℝ, (x1 + x2 = -2 * a) ∧ (x1 * x2 = 1) ∧ (x1 < 0) ∧ (x2 < 0)) ↔ (a ≥ 1) :=
by
  sorry

end range_of_a_l190_190050


namespace largest_of_7_consecutive_numbers_with_average_20_l190_190313

variable (n : ℤ) 

theorem largest_of_7_consecutive_numbers_with_average_20
  (h_avg : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6))/7 = 20) : 
  (n + 6) = 23 :=
by
  -- Placeholder for the actual proof
  sorry

end largest_of_7_consecutive_numbers_with_average_20_l190_190313


namespace function_decomposition_l190_190424

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ (a : ℝ) (f₁ f₂ : ℝ → ℝ), a > 0 ∧ (∀ x, f₁ x = f₁ (-x)) ∧ (∀ x, f₂ x = f₂ (2 * a - x)) ∧ (∀ x, f x = f₁ x + f₂ x) :=
sorry

end function_decomposition_l190_190424


namespace two_digit_number_representation_l190_190576

theorem two_digit_number_representation (x : ℕ) (h : x < 10) : 10 * x + 5 < 100 :=
by sorry

end two_digit_number_representation_l190_190576


namespace line_BC_l190_190079

noncomputable def Point := (ℝ × ℝ)
def A : Point := (-1, -4)
def l₁ := { p : Point | p.2 + 1 = 0 }
def l₂ := { p : Point | p.1 + p.2 + 1 = 0 }
def A' : Point := (-1, 2)
def A'' : Point := (3, 0)

theorem line_BC :
  ∃ (c₁ c₂ c₃ : ℝ), c₁ ≠ 0 ∨ c₂ ≠ 0 ∧
  ∀ (p : Point), (c₁ * p.1 + c₂ * p.2 + c₃ = 0) ↔ p ∈ { x | x = A ∨ x = A'' } :=
by sorry

end line_BC_l190_190079


namespace geometric_series_sum_l190_190720

def first_term : ℤ := 3
def common_ratio : ℤ := -2
def last_term : ℤ := -1536
def num_terms : ℕ := 10
def sum_of_series (a r : ℤ) (n : ℕ) : ℤ := a * ((r ^ n - 1) / (r - 1))

theorem geometric_series_sum :
  sum_of_series first_term common_ratio num_terms = -1023 := by
  sorry

end geometric_series_sum_l190_190720


namespace p_at_zero_l190_190845

-- Definitions according to given conditions
def p (x : ℝ) : ℝ := sorry  -- Polynomial of degree 6 with specific values

-- Given condition: Degree of polynomial
def degree_p : Prop := (∀ n : ℕ, (n ≤ 6) → p (3 ^ n) = 1 / 3 ^ n)

-- Theorem that needs to be proved
theorem p_at_zero : degree_p → p 0 = 6560 / 2187 := 
by
  sorry

end p_at_zero_l190_190845


namespace boxes_per_case_l190_190427

-- Define the conditions
def total_boxes : ℕ := 54
def total_cases : ℕ := 9

-- Define the result we want to prove
theorem boxes_per_case : total_boxes / total_cases = 6 := 
by sorry

end boxes_per_case_l190_190427


namespace polar_to_rectangular_coordinates_l190_190103

theorem polar_to_rectangular_coordinates 
  (r θ : ℝ) 
  (hr : r = 7) 
  (hθ : θ = 7 * Real.pi / 4) : 
  (r * Real.cos θ, r * Real.sin θ) = (7 * Real.sqrt 2 / 2, -7 * Real.sqrt 2 / 2) := 
by
  sorry

end polar_to_rectangular_coordinates_l190_190103


namespace rectangle_area_at_stage_8_l190_190731

-- Definitions based on conditions
def area_of_square (side_length : ℕ) : ℕ := side_length * side_length
def number_of_squares_in_stage (stage : ℕ) : ℕ := stage

-- The main theorem to prove
theorem rectangle_area_at_stage_8 : 
  area_of_square 4 * number_of_squares_in_stage 8 = 128 := by
  sorry

end rectangle_area_at_stage_8_l190_190731


namespace students_not_in_any_subject_l190_190653

theorem students_not_in_any_subject (total_students mathematics_students chemistry_students biology_students
  mathematics_chemistry_students chemistry_biology_students mathematics_biology_students all_three_students: ℕ)
  (h_total: total_students = 120) 
  (h_m: mathematics_students = 70)
  (h_c: chemistry_students = 50)
  (h_b: biology_students = 40)
  (h_mc: mathematics_chemistry_students = 30)
  (h_cb: chemistry_biology_students = 20)
  (h_mb: mathematics_biology_students = 10)
  (h_all: all_three_students = 5) :
  total_students - ((mathematics_students - mathematics_chemistry_students - mathematics_biology_students + all_three_students) +
    (chemistry_students - chemistry_biology_students - mathematics_chemistry_students + all_three_students) +
    (biology_students - chemistry_biology_students - mathematics_biology_students + all_three_students) +
    (mathematics_chemistry_students + chemistry_biology_students + mathematics_biology_students - 2 * all_three_students)) = 20 :=
by sorry

end students_not_in_any_subject_l190_190653


namespace scientific_notation_of_1300000_l190_190001

theorem scientific_notation_of_1300000 : 1300000 = 1.3 * 10^6 :=
by
  sorry

end scientific_notation_of_1300000_l190_190001


namespace adil_older_than_bav_by_732_days_l190_190085

-- Definitions based on the problem conditions
def adilBirthDate : String := "December 31, 2015"
def bavBirthDate : String := "January 1, 2018"

-- Main theorem statement 
theorem adil_older_than_bav_by_732_days :
    let daysIn2016 := 366
    let daysIn2017 := 365
    let transition := 1
    let totalDays := daysIn2016 + daysIn2017 + transition
    totalDays = 732 :=
by
    sorry

end adil_older_than_bav_by_732_days_l190_190085


namespace initial_deposit_l190_190008

theorem initial_deposit (A r : ℝ) (n t : ℕ) (hA : A = 169.40) 
  (hr : r = 0.20) (hn : n = 2) (ht : t = 1) :
  ∃ P : ℝ, P = 140 ∧ A = P * (1 + r / n)^(n * t) :=
by
  sorry

end initial_deposit_l190_190008


namespace order_of_reading_amounts_l190_190821

variable (a b c d : ℝ)

theorem order_of_reading_amounts (h1 : a + c = b + d) (h2 : a + b > c + d) (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c :=
by
  sorry

end order_of_reading_amounts_l190_190821


namespace race_time_diff_l190_190314

-- Define the speeds and race distance
def Malcolm_speed : ℕ := 5  -- in minutes per mile
def Joshua_speed : ℕ := 7   -- in minutes per mile
def Alice_speed : ℕ := 6    -- in minutes per mile
def race_distance : ℕ := 12 -- in miles

-- Calculate times
def Malcolm_time : ℕ := Malcolm_speed * race_distance
def Joshua_time : ℕ := Joshua_speed * race_distance
def Alice_time : ℕ := Alice_speed * race_distance

-- Lean 4 statement to prove the time differences
theorem race_time_diff :
  Joshua_time - Malcolm_time = 24 ∧ Alice_time - Malcolm_time = 12 := by
  sorry

end race_time_diff_l190_190314


namespace PointNegativeThreeTwo_l190_190922

def isInSecondQuadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem PointNegativeThreeTwo:
  isInSecondQuadrant (-3) 2 := by
  sorry

end PointNegativeThreeTwo_l190_190922


namespace snowball_game_l190_190794

theorem snowball_game (x y z : ℕ) (h : 5 * x + 4 * y + 3 * z = 12) : 
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end snowball_game_l190_190794


namespace sum_of_powers_modulo_seven_l190_190331

theorem sum_of_powers_modulo_seven :
  ((1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7) % 7) = 1 := by
  sorry

end sum_of_powers_modulo_seven_l190_190331


namespace three_digit_number_formed_by_1198th_1200th_digits_l190_190277

def albertSequenceDigit (n : ℕ) : ℕ :=
  -- Define the nth digit in Albert's sequence
  sorry

theorem three_digit_number_formed_by_1198th_1200th_digits :
  let d1198 := albertSequenceDigit 1198
  let d1199 := albertSequenceDigit 1199
  let d1200 := albertSequenceDigit 1200
  (d1198 * 100 + d1199 * 10 + d1200) = 220 :=
by
  sorry

end three_digit_number_formed_by_1198th_1200th_digits_l190_190277


namespace petya_mistake_l190_190000

theorem petya_mistake (x : ℝ) (h : x - x / 10 = 19.71) : x = 21.9 := 
  sorry

end petya_mistake_l190_190000


namespace range_of_m_l190_190027

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ (Set.Ioc 0 2) then 2^x - 1 else sorry

def g (x m : ℝ) : ℝ :=
x^2 - 2*x + m

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-2:ℝ) 2, f (-x) = -f x) ∧
  (∀ x ∈ Set.Ioc (0:ℝ) 2, f x = 2^x - 1) ∧
  (∀ x1 ∈ Set.Icc (-2:ℝ) 2, ∃ x2 ∈ Set.Icc (-2:ℝ) 2, g x2 m = f x1) 
  → -5 ≤ m ∧ m ≤ -2 :=
sorry

end range_of_m_l190_190027


namespace inequality_range_of_k_l190_190145

theorem inequality_range_of_k 
  (a b k : ℝ)
  (h : ∀ a b : ℝ, a^2 + b^2 ≥ 2 * k * a * b) : k ∈ Set.Icc (-1 : ℝ) (1 : ℝ) :=
by
  sorry

end inequality_range_of_k_l190_190145


namespace range_of_a_l190_190078

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 2

-- Prove that if f(x) is decreasing on ℝ, then a must be less than or equal to -3
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (3 * a * x^2 + 6 * x - 1) < 0 ) → a ≤ -3 :=
sorry

end range_of_a_l190_190078


namespace find_m_l190_190207

theorem find_m (m : ℝ) 
    (h1 : ∃ (m: ℝ), ∀ x y : ℝ, x - m * y + 2 * m = 0) 
    (h2 : ∃ (m: ℝ), ∀ x y : ℝ, x + 2 * y - m = 0) 
    (perpendicular : (1/m) * (-1/2) = -1) : m = 1/2 :=
sorry

end find_m_l190_190207


namespace intersection_of_A_and_B_l190_190650

def setA : Set ℝ := { x | (x - 3) * (x + 1) ≥ 0 }
def setB : Set ℝ := { x | x < -4/5 }

theorem intersection_of_A_and_B : setA ∩ setB = { x | x ≤ -1 } :=
  sorry

end intersection_of_A_and_B_l190_190650


namespace no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime_l190_190487

theorem no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime :
  ¬∃ n : ℕ, 2 ≤ n ∧ Nat.Prime (n^4 + n^2 + 1) :=
sorry

end no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime_l190_190487


namespace find_x_l190_190854

-- Definitions to capture angles and triangle constraints
def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

def perpendicular (A B : ℝ) : Prop := A + B = 90

-- Given conditions
axiom angle_ABC : ℝ
axiom angle_BAC : ℝ
axiom angle_BCA : ℝ
axiom angle_DCE : ℝ
axiom angle_x : ℝ

-- Specific values for the angles provided in the problem
axiom angle_ABC_is_70 : angle_ABC = 70
axiom angle_BAC_is_50 : angle_BAC = 50

-- Angle BCA in triangle ABC
axiom angle_sum_ABC : angle_sum_triangle angle_ABC angle_BAC angle_BCA

-- Conditional relationships in triangle CDE
axiom angle_DCE_equals_BCA : angle_DCE = angle_BCA
axiom angle_sum_CDE : perpendicular angle_DCE angle_x

-- The theorem we need to prove
theorem find_x : angle_x = 30 := sorry

end find_x_l190_190854


namespace circle_center_radius_sum_l190_190387

theorem circle_center_radius_sum (u v s : ℝ) (h1 : (x + 4)^2 + (y - 1)^2 = 13)
    (h2 : (u, v) = (-4, 1)) (h3 : s = Real.sqrt 13) : 
    u + v + s = -3 + Real.sqrt 13 :=
by
  sorry

end circle_center_radius_sum_l190_190387


namespace minimum_value_of_v_l190_190217

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

noncomputable def g (x u v : ℝ) : ℝ := (x - u)^3 - 3 * (x - u) - v

theorem minimum_value_of_v (u v : ℝ) (h_pos_u : u > 0) :
  ∀ u > 0, ∀ x : ℝ, f x = g x u v → v ≥ 4 :=
by
  sorry

end minimum_value_of_v_l190_190217


namespace base_of_1987_with_digit_sum_25_l190_190147

theorem base_of_1987_with_digit_sum_25 (b a c : ℕ) (h₀ : a * b^2 + b * b + c = 1987)
(h₁ : a + b + c = 25) (h₂ : 1 ≤ b ∧ b ≤ 45) : b = 19 :=
sorry

end base_of_1987_with_digit_sum_25_l190_190147


namespace intersection_product_distance_eq_eight_l190_190305

noncomputable def parametricCircle : ℝ → ℝ × ℝ :=
  λ θ => (4 * Real.cos θ, 4 * Real.sin θ)

noncomputable def parametricLine : ℝ → ℝ × ℝ :=
  λ t => (2 + (1 / 2) * t, 2 + (Real.sqrt 3 / 2) * t)

theorem intersection_product_distance_eq_eight :
  ∀ θ t,
    let (x1, y1) := parametricCircle θ
    let (x2, y2) := parametricLine t
    (x1^2 + y1^2 = 16) ∧ (x2 = x1 ∧ y2 = y1) →
    ∃ t1 t2,
      x1 = 2 + (1 / 2) * t1 ∧ y1 = 2 + (Real.sqrt 3 / 2) * t1 ∧
      x1 = 2 + (1 / 2) * t2 ∧ y1 = 2 + (Real.sqrt 3 / 2) * t2 ∧
      (t1 * t2 = -8) ∧ (|t1 * t2| = 8) := 
by
  intros θ t
  dsimp only
  intro h
  sorry

end intersection_product_distance_eq_eight_l190_190305


namespace problem_statement_l190_190830

variable {a b c d k : ℝ}

theorem problem_statement (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h_pos : 0 < k)
    (h_sum_ab : a + b = k)
    (h_sum_cd : c + d = k^2)
    (h_roots1 : ∀ x, x^2 - 4*a*x - 5*b = 0 → x = c ∨ x = d)
    (h_roots2 : ∀ x, x^2 - 4*c*x - 5*d = 0 → x = a ∨ x = b) : 
    a + b + c + d = k + k^2 :=
sorry

end problem_statement_l190_190830


namespace find_angle_B_l190_190614

noncomputable def triangle_sides_and_angles 
(a b c : ℝ) (A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

noncomputable def vectors_parallel 
(A B C a b c : ℝ) : Prop :=
  (Real.sin B - Real.sin A) / Real.sin C = (Real.sqrt 3 * a + c) / (a + b)

theorem find_angle_B (A B C a b c : ℝ)
  (h_triangle : triangle_sides_and_angles a b c A B C)
  (h_parallel : vectors_parallel A B C a b c) :
  B = 5 * Real.pi / 6 :=
sorry

end find_angle_B_l190_190614


namespace carnival_game_ratio_l190_190543

theorem carnival_game_ratio (L W : ℕ) (h_ratio : 4 * L = W) (h_lost : L = 7) : W = 28 :=
by {
  sorry
}

end carnival_game_ratio_l190_190543


namespace domain_of_f_l190_190223

-- Define the function f(x) = 1/(x+1) + ln(x)
noncomputable def f (x : ℝ) : ℝ := (1 / (x + 1)) + Real.log x

-- The domain of the function is all x such that x > 0
theorem domain_of_f :
  ∀ x : ℝ, (x > 0) ↔ (f x = (1 / (x + 1)) + Real.log x) := 
by sorry

end domain_of_f_l190_190223


namespace simplify_polynomial_l190_190803

theorem simplify_polynomial (x : ℝ) (A B C D : ℝ) :
  (y = (x^3 + 12 * x^2 + 47 * x + 60) / (x + 3)) →
  (y = A * x^2 + B * x + C) →
  x ≠ D →
  A = 1 ∧ B = 9 ∧ C = 20 ∧ D = -3 :=
by
  sorry

end simplify_polynomial_l190_190803


namespace point_in_which_quadrant_l190_190773

theorem point_in_which_quadrant (x y : ℝ) (h1 : y = 2 * x + 3) (h2 : abs x = abs y) :
  (x < 0 ∧ y < 0) ∨ (x < 0 ∧ y > 0) :=
by
  -- Proof omitted
  sorry

end point_in_which_quadrant_l190_190773


namespace ruel_usable_stamps_l190_190273

def totalStamps (books10 books15 books25 books30 : ℕ) (stamps10 stamps15 stamps25 stamps30 : ℕ) : ℕ :=
  books10 * stamps10 + books15 * stamps15 + books25 * stamps25 + books30 * stamps30

def damagedStamps (damaged25 damaged30 : ℕ) : ℕ :=
  damaged25 + damaged30

def usableStamps (books10 books15 books25 books30 stamps10 stamps15 stamps25 stamps30 damaged25 damaged30 : ℕ) : ℕ :=
  totalStamps books10 books15 books25 books30 stamps10 stamps15 stamps25 stamps30 - damagedStamps damaged25 damaged30

theorem ruel_usable_stamps :
  usableStamps 4 6 3 2 10 15 25 30 5 3 = 257 := by
  sorry

end ruel_usable_stamps_l190_190273


namespace books_distribution_l190_190357

noncomputable def distribution_ways : ℕ :=
  let books := 5
  let people := 4
  let combination := Nat.choose books 2
  let arrangement := Nat.factorial people
  combination * arrangement ^ people

theorem books_distribution : distribution_ways = 240 := by
  sorry

end books_distribution_l190_190357


namespace seating_arrangements_family_van_correct_l190_190339

noncomputable def num_seating_arrangements (parents : Fin 2) (children : Fin 3) : Nat :=
  let perm3_2 := Nat.factorial 3 / Nat.factorial (3 - 2)
  2 * 1 * perm3_2

theorem seating_arrangements_family_van_correct :
  num_seating_arrangements 2 3 = 12 :=
by
  sorry

end seating_arrangements_family_van_correct_l190_190339


namespace volume_conversion_l190_190275

-- Define the given conditions
def V_feet : ℕ := 216
def C_factor : ℕ := 27

-- State the theorem to prove
theorem volume_conversion : V_feet / C_factor = 8 :=
  sorry

end volume_conversion_l190_190275


namespace radius_of_circle_eq_zero_l190_190707

theorem radius_of_circle_eq_zero :
  ∀ x y: ℝ, (x^2 + 8 * x + y^2 - 10 * y + 41 = 0) → (0 : ℝ) = 0 :=
by
  intros x y h
  sorry

end radius_of_circle_eq_zero_l190_190707


namespace sample_older_employees_count_l190_190678

-- Definitions of known quantities
def N := 400
def N_older := 160
def N_no_older := 240
def n := 50

-- The proof statement showing that the number of employees older than 45 in the sample equals 20
theorem sample_older_employees_count : 
  let proportion_older := (N_older:ℝ) / (N:ℝ)
  let n_older := proportion_older * (n:ℝ)
  n_older = 20 := by
  sorry

end sample_older_employees_count_l190_190678


namespace tourists_escape_l190_190083

theorem tourists_escape (T : ℕ) (hT : T = 10) (hats : Fin T → Bool) (could_see : ∀ (i : Fin T), Fin (i) → Bool) :
  ∃ strategy : (Fin T → Bool), (∀ (i : Fin T), (strategy i = hats i) ∨ (strategy i ≠ hats i)) →
  (∀ (i : Fin T), (∀ (j : Fin T), i < j → strategy i = hats i) → ∃ count : ℕ, count ≥ 9 ∧ ∀ (i : Fin T), count ≥ i → strategy i = hats i) := sorry

end tourists_escape_l190_190083


namespace find_positive_real_x_l190_190104

noncomputable def positive_solution :=
  ∃ (x : ℝ), (1/3) * (4 * x^2 - 2) = (x^2 - 75 * x - 15) * (x^2 + 50 * x + 10) ∧ x > 0

theorem find_positive_real_x :
  positive_solution ↔ ∃ (x : ℝ), x = (75 + Real.sqrt 5693) / 2 :=
by sorry

end find_positive_real_x_l190_190104


namespace john_initial_investment_in_alpha_bank_is_correct_l190_190060

-- Definition of the problem conditions
def initial_investment : ℝ := 2000
def alpha_rate : ℝ := 0.04
def beta_rate : ℝ := 0.06
def final_amount : ℝ := 2398.32
def years : ℕ := 3

-- Alpha Bank growth factor after 3 years
def alpha_growth_factor : ℝ := (1 + alpha_rate) ^ years

-- Beta Bank growth factor after 3 years
def beta_growth_factor : ℝ := (1 + beta_rate) ^ years

-- The main theorem
theorem john_initial_investment_in_alpha_bank_is_correct (x : ℝ) 
  (hx : x * alpha_growth_factor + (initial_investment - x) * beta_growth_factor = final_amount) : 
  x = 246.22 :=
sorry

end john_initial_investment_in_alpha_bank_is_correct_l190_190060


namespace find_value_l190_190728

-- Defining the sequence a_n, assuming all terms are positive
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = a n * r

-- Definition to capture the given condition a_2 * a_4 = 4
def condition (a : ℕ → ℝ) : Prop :=
  a 2 * a 4 = 4

-- The main statement
theorem find_value (a : ℕ → ℝ) (h_seq : is_geometric_sequence a) (h_cond : condition a) : 
  a 1 * a 5 + a 3 = 6 := 
by 
  sorry

end find_value_l190_190728


namespace sum_of_first_20_terms_arithmetic_sequence_l190_190206

theorem sum_of_first_20_terms_arithmetic_sequence 
  (a : ℕ → ℤ)
  (h_arith : ∃ d : ℤ, ∀ n, a n = a 0 + n * d)
  (h_sum_first_three : a 0 + a 1 + a 2 = -24)
  (h_sum_eighteen_nineteen_twenty : a 17 + a 18 + a 19 = 78) :
  (20 / 2 * (a 0 + (a 0 + 19 * d))) = 180 :=
by
  sorry

end sum_of_first_20_terms_arithmetic_sequence_l190_190206


namespace find_k_l190_190218

theorem find_k (k m : ℝ) : (m^2 - 8*m) ∣ (m^3 - k*m^2 - 24*m + 16) → k = 8 := by
  sorry

end find_k_l190_190218


namespace C_finishes_work_in_days_l190_190450

theorem C_finishes_work_in_days :
  (∀ (unit : ℝ) (A B C combined: ℝ),
    combined = 1 / 4 ∧
    A = 1 / 12 ∧
    B = 1 / 24 ∧
    combined = A + B + 1 / C) → 
    C = 8 :=
  sorry

end C_finishes_work_in_days_l190_190450


namespace largest_prime_divisor_in_range_l190_190451

theorem largest_prime_divisor_in_range (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  ∃ p, Prime p ∧ p ≤ Int.floor (Real.sqrt n) ∧ 
  (∀ q, Prime q ∧ q ≤ Int.floor (Real.sqrt n) → q ≤ p) :=
sorry

end largest_prime_divisor_in_range_l190_190451


namespace no_integer_solutions_exist_l190_190713

theorem no_integer_solutions_exist (n m : ℤ) : 
  (n ^ 2 - m ^ 2 = 250) → false := 
sorry 

end no_integer_solutions_exist_l190_190713


namespace subset_div_chain_l190_190711

theorem subset_div_chain (m n : ℕ) (h_m : m > 0) (h_n : n > 0) (S : Finset ℕ) (hS : S.card = (2^m - 1) * n + 1) (hS_subset : S ⊆ Finset.range (2^(m) * n + 1)) :
  ∃ (a : Fin (m+1) → ℕ), (∀ i, a i ∈ S) ∧ (∀ k : ℕ, k < m → a k ∣ a (k + 1)) :=
sorry

end subset_div_chain_l190_190711


namespace possible_values_of_expression_l190_190615

theorem possible_values_of_expression (x y : ℝ) (hxy : x + 2 * y = 2) (hx_pos : x > 0) (hy_pos : y > 0) :
  ∃ v, v = 21 / 4 ∧ (1 / x + 2 / y) = v :=
sorry

end possible_values_of_expression_l190_190615


namespace max_true_statements_l190_190300

theorem max_true_statements :
  ∃ x : ℝ, 
  (0 < x ∧ x < 1) ∧ -- Statement 4
  (0 < x^3 ∧ x^3 < 1) ∧ -- Statement 1
  (0 < x - x^3 ∧ x - x^3 < 1) ∧ -- Statement 5
  ¬(x^3 > 1) ∧ -- Not Statement 2
  ¬(-1 < x ∧ x < 0) := -- Not Statement 3
sorry

end max_true_statements_l190_190300


namespace read_both_books_l190_190240

theorem read_both_books (B S K N : ℕ) (TOTAL : ℕ)
  (h1 : S = 1/4 * 72)
  (h2 : K = 5/8 * 72)
  (h3 : N = (S - B) - 1)
  (h4 : TOTAL = 72)
  (h5 : TOTAL = (S - B) + (K - B) + B + N)
  : B = 8 :=
by
  sorry

end read_both_books_l190_190240


namespace sqrt_37_range_l190_190570

theorem sqrt_37_range : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 :=
by
  sorry

end sqrt_37_range_l190_190570


namespace integer_roots_of_quadratic_l190_190517

theorem integer_roots_of_quadratic (b : ℤ) :
  (∃ x : ℤ, x^2 + 4 * x + b = 0) ↔ b = -12 ∨ b = -5 ∨ b = 3 ∨ b = 4 :=
sorry

end integer_roots_of_quadratic_l190_190517


namespace g_at_1_l190_190664

variable (g : ℝ → ℝ)

theorem g_at_1 (h : ∀ x : ℝ, g (2 * x - 5) = 3 * x + 9) : g 1 = 18 := by
  sorry

end g_at_1_l190_190664


namespace solve_system_eq_l190_190880

theorem solve_system_eq (x y : ℝ) (h1 : 2 * x - y = 3) (h2 : 3 * x + 2 * y = 8) :
  x = 2 ∧ y = 1 :=
by
  sorry

end solve_system_eq_l190_190880


namespace positive_integers_solution_l190_190741

theorem positive_integers_solution :
  ∀ (m n : ℕ), 0 < m ∧ 0 < n ∧ (3 ^ m - 2 ^ n = -1 ∨ 3 ^ m - 2 ^ n = 5 ∨ 3 ^ m - 2 ^ n = 7) ↔
  (m, n) = (0, 1) ∨ (m, n) = (2, 1) ∨ (m, n) = (1, 2) ∨ (m, n) = (2, 2) :=
by
  sorry

end positive_integers_solution_l190_190741


namespace g_does_not_pass_second_quadrant_l190_190228

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x+5) + 4

def M : ℝ × ℝ := (-5, 5)

noncomputable def g (x : ℝ) : ℝ := -5 + (5 : ℝ)^(x)

theorem g_does_not_pass_second_quadrant (a : ℝ) (x : ℝ) 
  (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hM : f a (-5) = 5) : 
  ∀ x < 0, g x < 0 :=
by
  sorry

end g_does_not_pass_second_quadrant_l190_190228


namespace sum_of_cube_faces_l190_190265

theorem sum_of_cube_faces :
  ∃ (a b c d e f : ℕ), 
    (a = 12) ∧ 
    (b = a + 3) ∧ 
    (c = b + 3) ∧ 
    (d = c + 3) ∧ 
    (e = d + 3) ∧ 
    (f = e + 3) ∧ 
    (a + f = 39) ∧ 
    (b + e = 39) ∧ 
    (c + d = 39) ∧ 
    (a + b + c + d + e + f = 117) :=
by
  let a := 12
  let b := a + 3
  let c := b + 3
  let d := c + 3
  let e := d + 3
  let f := e + 3
  have h1 : a + f = 39 := sorry
  have h2 : b + e = 39 := sorry
  have h3 : c + d = 39 := sorry
  have sum : a + b + c + d + e + f = 117 := sorry
  exact ⟨a, b, c, d, e, f, rfl, rfl, rfl, rfl, rfl, rfl, h1, h2, h3, sum⟩

end sum_of_cube_faces_l190_190265


namespace coeff_x3_in_product_l190_190271

open Polynomial

noncomputable def p : Polynomial ℤ := 3 * X^3 + 2 * X^2 + 5 * X + 3
noncomputable def q : Polynomial ℤ := 4 * X^3 + 5 * X^2 + 6 * X + 8

theorem coeff_x3_in_product :
  (p * q).coeff 3 = 61 :=
by sorry

end coeff_x3_in_product_l190_190271


namespace determinant_identity_l190_190343

variable (a b : ℝ)

theorem determinant_identity :
  Matrix.det ![
      ![1, Real.sin (a - b), Real.sin a],
      ![Real.sin (a - b), 1, Real.sin b],
      ![Real.sin a, Real.sin b, 1]
  ] = 0 :=
by sorry

end determinant_identity_l190_190343


namespace common_roots_product_sum_l190_190648

theorem common_roots_product_sum (C D u v w t p q r : ℝ) (huvw : u^3 + C * u - 20 = 0) (hvw : v^3 + C * v - 20 = 0)
  (hw: w^3 + C * w - 20 = 0) (hut: t^3 + D * t^2 - 40 = 0) (hvw: v^3 + D * v^2 - 40 = 0) 
  (hu: u^3 + D * u^2 - 40 = 0) (h1: u + v + w = 0) (h2: u * v * w = 20) 
  (h3: u * v + u * t + v * t = 0) (h4: u * v * t = 40) :
  p = 4 → q = 3 → r = 5 → p + q + r = 12 :=
by sorry

end common_roots_product_sum_l190_190648


namespace max_ab_upper_bound_l190_190110

noncomputable def circle_center_coords : ℝ × ℝ :=
  let center_x := -1
  let center_y := 2
  (center_x, center_y)

noncomputable def max_ab_value (a b : ℝ) : ℝ :=
  if a = 1 - 2 * b then a * b else 0

theorem max_ab_upper_bound :
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + 2*p.1 - 4*p.2 + 1 = 0}
  let line_cond : ℝ × ℝ := (-1, 2)
  (circle_center_coords = line_cond) →
  (∀ a b : ℝ, max_ab_value a b ≤ 1 / 8) :=
by
  intro circle line_cond h
  -- Proof is omitted as per instruction
  sorry

end max_ab_upper_bound_l190_190110


namespace interest_rate_for_lending_l190_190600

def simple_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℕ :=
  (P * R * T) / 100

theorem interest_rate_for_lending :
  ∀ (P T R_b G R_l : ℕ),
  P = 20000 →
  T = 6 →
  R_b = 8 →
  G = 200 →
  simple_interest P R_b T + G * T = simple_interest P R_l T →
  R_l = 9 :=
by
  intros P T R_b G R_l
  sorry

end interest_rate_for_lending_l190_190600


namespace clarinet_cost_correct_l190_190800

noncomputable def total_spent : ℝ := 141.54
noncomputable def song_book_cost : ℝ := 11.24
noncomputable def clarinet_cost : ℝ := total_spent - song_book_cost

theorem clarinet_cost_correct : clarinet_cost = 130.30 :=
by
  sorry

end clarinet_cost_correct_l190_190800


namespace solve_fraction_problem_l190_190916

theorem solve_fraction_problem (n : ℝ) (h : (4 + n) / (7 + n) = 7 / 9) : n = 13 / 2 :=
by
  sorry

end solve_fraction_problem_l190_190916


namespace intercepts_of_line_l190_190460

-- Define the given line equation
def line_eq (x y : ℝ) : Prop := x / 4 - y / 3 = 1

-- Define the intercepts
def intercepts (x_intercept y_intercept : ℝ) : Prop :=
  (line_eq x_intercept 0) ∧ (line_eq 0 y_intercept)

-- The problem statement: proving the values of intercepts
theorem intercepts_of_line :
  intercepts 4 (-3) :=
by
  sorry

end intercepts_of_line_l190_190460


namespace parabola_intersection_points_l190_190829

theorem parabola_intersection_points :
  let parabola1 := λ x : ℝ => 4*x^2 + 3*x - 1
  let parabola2 := λ x : ℝ => x^2 + 8*x + 7
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ = -4/3 ∧ y₁ = -17/9 ∧
                        x₂ = 2 ∧ y₂ = 27 ∧
                        parabola1 x₁ = y₁ ∧ 
                        parabola2 x₁ = y₁ ∧
                        parabola1 x₂ = y₂ ∧
                        parabola2 x₂ = y₂ :=
by {
  sorry
}

end parabola_intersection_points_l190_190829


namespace fruit_weights_determined_l190_190566

noncomputable def fruit_weight_configuration : Prop :=
  let weights : List ℕ := [100, 150, 170, 200, 280]
  let mandarin := 100
  let apple := 150
  let banana := 170
  let peach := 200
  let orange := 280
  (peach < orange) ∧
  (apple < banana ∧ banana < peach) ∧
  (mandarin < banana) ∧
  (apple + banana > orange)
  
theorem fruit_weights_determined :
  fruit_weight_configuration :=
by
  sorry

end fruit_weights_determined_l190_190566


namespace phone_answered_before_fifth_ring_l190_190412

theorem phone_answered_before_fifth_ring:
  (0.1 + 0.2 + 0.25 + 0.25 = 0.8) :=
by
  sorry

end phone_answered_before_fifth_ring_l190_190412


namespace sin_cos_value_l190_190743

variable (x : ℝ)

theorem sin_cos_value (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
sorry

end sin_cos_value_l190_190743


namespace Yoongi_has_fewest_apples_l190_190975

def Jungkook_apples : Nat := 6 * 3
def Yoongi_apples : Nat := 4
def Yuna_apples : Nat := 5

theorem Yoongi_has_fewest_apples :
  Yoongi_apples < Jungkook_apples ∧ Yoongi_apples < Yuna_apples :=
by
  sorry

end Yoongi_has_fewest_apples_l190_190975


namespace min_value_of_a_plus_b_l190_190416

theorem min_value_of_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) : a + b = 4 :=
sorry

end min_value_of_a_plus_b_l190_190416


namespace mass_percentage_C_in_CO_l190_190301

noncomputable def atomic_mass_C : ℚ := 12.01
noncomputable def atomic_mass_O : ℚ := 16.00
noncomputable def molecular_mass_CO : ℚ := atomic_mass_C + atomic_mass_O

theorem mass_percentage_C_in_CO : (atomic_mass_C / molecular_mass_CO) * 100 = 42.88 :=
by
  have atomic_mass_C_div_total : atomic_mass_C / molecular_mass_CO = 12.01 / 28.01 := sorry
  have mass_percentage : (atomic_mass_C / molecular_mass_CO) * 100 = 42.88 := sorry
  exact mass_percentage

end mass_percentage_C_in_CO_l190_190301


namespace seeds_total_l190_190294

theorem seeds_total (wednesday_seeds thursday_seeds : ℕ) (h_wed : wednesday_seeds = 20) (h_thu : thursday_seeds = 2) : (wednesday_seeds + thursday_seeds) = 22 := by
  sorry

end seeds_total_l190_190294


namespace angle_bisector_correct_length_l190_190322

-- Define the isosceles triangle with the given conditions
structure IsoscelesTriangle :=
  (base : ℝ)
  (lateral : ℝ)
  (is_isosceles : lateral = 20 ∧ base = 5)

-- Define the problem of finding the angle bisector
noncomputable def angle_bisector_length (tri : IsoscelesTriangle) : ℝ :=
  6

-- The main theorem to state the problem
theorem angle_bisector_correct_length (tri : IsoscelesTriangle) : 
  angle_bisector_length tri = 6 :=
by
  -- We state the theorem, skipping the proof (sorry)
  sorry

end angle_bisector_correct_length_l190_190322


namespace chili_problem_l190_190046

def cans_of_chili (x y z : ℕ) : Prop := x + 2 * y + z = 6

def percentage_more_tomatoes_than_beans (x y z : ℕ) : ℕ :=
  100 * (z - 2 * y) / (2 * y)

theorem chili_problem (x y z : ℕ) (h1 : cans_of_chili x y z) (h2 : x = 1) (h3 : y = 1) : 
  percentage_more_tomatoes_than_beans x y z = 50 :=
by
  sorry

end chili_problem_l190_190046


namespace original_price_double_value_l190_190063

theorem original_price_double_value :
  ∃ (P : ℝ), P + 0.30 * P = 351 ∧ 2 * P = 540 :=
by
  sorry

end original_price_double_value_l190_190063


namespace quadratic_vertex_coords_l190_190685

theorem quadratic_vertex_coords :
  ∀ x : ℝ, (y = (x-2)^2 - 1) → (2, -1) = (2, -1) :=
by
  sorry

end quadratic_vertex_coords_l190_190685


namespace cost_per_ounce_l190_190913

theorem cost_per_ounce (total_cost : ℕ) (num_ounces : ℕ) (h1 : total_cost = 84) (h2 : num_ounces = 12) : (total_cost / num_ounces) = 7 :=
by
  sorry

end cost_per_ounce_l190_190913


namespace find_fourth_power_sum_l190_190408

theorem find_fourth_power_sum (a b c : ℝ) 
    (h1 : a + b + c = 2) 
    (h2 : a^2 + b^2 + c^2 = 3) 
    (h3 : a^3 + b^3 + c^3 = 4) : 
    a^4 + b^4 + c^4 = 7.833 :=
sorry

end find_fourth_power_sum_l190_190408


namespace sides_of_nth_hexagon_l190_190605

-- Definition of the arithmetic sequence condition.
def first_term : ℕ := 6
def common_difference : ℕ := 5

-- The function representing the n-th term of the sequence.
def num_sides (n : ℕ) : ℕ := first_term + (n - 1) * common_difference

-- Now, we state the theorem that the n-th term equals 5n + 1.
theorem sides_of_nth_hexagon (n : ℕ) : num_sides n = 5 * n + 1 := by
  sorry

end sides_of_nth_hexagon_l190_190605


namespace employee_b_payment_l190_190016

theorem employee_b_payment (total_payment : ℝ) (A_ratio : ℝ) (payment_B : ℝ) : 
  total_payment = 550 ∧ A_ratio = 1.2 ∧ total_payment = payment_B + A_ratio * payment_B → payment_B = 250 := 
by
  sorry

end employee_b_payment_l190_190016


namespace brad_started_after_maxwell_l190_190928

theorem brad_started_after_maxwell :
  ∀ (distance maxwell_speed brad_speed maxwell_time : ℕ),
  distance = 94 →
  maxwell_speed = 4 →
  brad_speed = 6 →
  maxwell_time = 10 →
  (distance - maxwell_speed * maxwell_time) / brad_speed = 9 := 
by
  intros distance maxwell_speed brad_speed maxwell_time h_dist h_m_speed h_b_speed h_m_time
  sorry

end brad_started_after_maxwell_l190_190928


namespace remainder_eq_159_l190_190402

def x : ℕ := 2^40
def numerator : ℕ := 2^160 + 160
def denominator : ℕ := 2^80 + 2^40 + 1

theorem remainder_eq_159 : (numerator % denominator) = 159 := 
by {
  -- Proof will be filled in here.
  sorry
}

end remainder_eq_159_l190_190402


namespace soccer_league_points_l190_190835

structure Team :=
  (name : String)
  (regular_wins : ℕ)
  (losses : ℕ)
  (draws : ℕ)
  (bonus_wins : ℕ)

def total_points (t : Team) : ℕ :=
  3 * t.regular_wins + t.draws + 2 * t.bonus_wins

def Team_Soccer_Stars : Team :=
  { name := "Team Soccer Stars", regular_wins := 18, losses := 5, draws := 7, bonus_wins := 6 }

def Lightning_Strikers : Team :=
  { name := "Lightning Strikers", regular_wins := 15, losses := 8, draws := 7, bonus_wins := 5 }

def Goal_Grabbers : Team :=
  { name := "Goal Grabbers", regular_wins := 21, losses := 5, draws := 4, bonus_wins := 4 }

def Clever_Kickers : Team :=
  { name := "Clever Kickers", regular_wins := 11, losses := 10, draws := 9, bonus_wins := 2 }

theorem soccer_league_points :
  total_points Team_Soccer_Stars = 73 ∧
  total_points Lightning_Strikers = 62 ∧
  total_points Goal_Grabbers = 75 ∧
  total_points Clever_Kickers = 46 ∧
  [Goal_Grabbers, Team_Soccer_Stars, Lightning_Strikers, Clever_Kickers].map total_points =
  [75, 73, 62, 46] := 
by
  sorry

end soccer_league_points_l190_190835


namespace subset_implies_all_elements_l190_190598

variable {U : Type}

theorem subset_implies_all_elements (P Q : Set U) (hPQ : P ⊆ Q) (hP_nonempty : P ≠ ∅) (hQ_nonempty : Q ≠ ∅) :
  ∀ x ∈ P, x ∈ Q :=
by 
  sorry

end subset_implies_all_elements_l190_190598


namespace profit_percentage_is_ten_l190_190573

-- Definitions based on conditions
def cost_price := 500
def selling_price := 550

-- Defining the profit percentage
def profit := selling_price - cost_price
def profit_percentage := (profit / cost_price) * 100

-- The proof that the profit percentage is 10
theorem profit_percentage_is_ten : profit_percentage = 10 :=
by
  -- Using the definitions given
  sorry

end profit_percentage_is_ten_l190_190573


namespace sum_of_five_integers_l190_190234

theorem sum_of_five_integers :
  ∃ (n m : ℕ), (n * (n + 1) = 336) ∧ ((m - 1) * m * (m + 1) = 336) ∧ ((n + (n + 1) + (m - 1) + m + (m + 1)) = 51) := 
sorry

end sum_of_five_integers_l190_190234


namespace transformed_center_coordinates_l190_190729

-- Define the original center of the circle
def center_initial : ℝ × ℝ := (3, -4)

-- Define the function for reflection across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the function for translation by a certain number of units up
def translate_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

-- Define the problem statement
theorem transformed_center_coordinates :
  translate_up (reflect_x_axis center_initial) 5 = (3, 9) :=
by
  sorry

end transformed_center_coordinates_l190_190729


namespace probability_of_selection_l190_190875

/-- A school selects 80 students for a discussion from a total of 883 students. First, 3 people are eliminated using simple random sampling, and then 80 are selected from the remaining 880 using systematic sampling. Prove that the probability of each person being selected is 80/883. -/
theorem probability_of_selection (total_students : ℕ) (students_eliminated : ℕ) (students_selected : ℕ) 
  (h_total : total_students = 883) (h_eliminated : students_eliminated = 3) (h_selected : students_selected = 80) :
  ((total_students - students_eliminated) * students_selected) / (total_students * (total_students - students_eliminated)) = 80 / 883 :=
by
  sorry

end probability_of_selection_l190_190875


namespace find_x_l190_190486

theorem find_x (x y : ℕ) (h1 : y = 144) (h2 : x^3 * 6^2 / 432 = y) : x = 12 := 
by
  sorry

end find_x_l190_190486


namespace operation_result_l190_190492

-- Define x and the operations
def x : ℕ := 40

-- Define the operation sequence
def operation (y : ℕ) : ℕ :=
  let step1 := y / 4
  let step2 := step1 * 5
  let step3 := step2 + 10
  let step4 := step3 - 12
  step4

-- The statement we need to prove
theorem operation_result : operation x = 48 := by
  sorry

end operation_result_l190_190492


namespace tom_beach_days_l190_190817

theorem tom_beach_days (total_seashells days_seashells : ℕ) (found_each_day total_found : ℕ) 
    (h1 : found_each_day = 7) (h2 : total_found = 35) : total_found / found_each_day = 5 := 
by 
  sorry

end tom_beach_days_l190_190817


namespace inequality_holds_l190_190798

variables {a b c : ℝ}

theorem inequality_holds (h1 : c < b) (h2 : b < a) (h3 : ac < 0) : ab > ac :=
sorry

end inequality_holds_l190_190798


namespace second_piece_weight_l190_190812

theorem second_piece_weight (w1 : ℝ) (s1 : ℝ) (s2 : ℝ) (w2 : ℝ) :
  (s1 = 4) → (w1 = 16) → (s2 = 6) → w2 = w1 * (s2^2 / s1^2) → w2 = 36 :=
by
  intro h_s1 h_w1 h_s2 h_w2
  rw [h_s1, h_w1, h_s2] at h_w2
  norm_num at h_w2
  exact h_w2

end second_piece_weight_l190_190812


namespace factorization_of_polynomial_l190_190221

theorem factorization_of_polynomial (x : ℝ) : 12 * x^2 - 40 * x + 25 = (2 * Real.sqrt 3 * x - 5)^2 :=
  sorry

end factorization_of_polynomial_l190_190221


namespace cuboid_volume_l190_190454

theorem cuboid_volume (P h : ℝ) (P_eq : P = 32) (h_eq : h = 9) :
  ∃ (s : ℝ), 4 * s = P ∧ s * s * h = 576 :=
by
  sorry

end cuboid_volume_l190_190454


namespace vectors_not_coplanar_l190_190194

def vector_a : Fin 3 → ℤ := ![1, 5, 2]
def vector_b : Fin 3 → ℤ := ![-1, 1, -1]
def vector_c : Fin 3 → ℤ := ![1, 1, 1]

def scalar_triple_product (a b c : Fin 3 → ℤ) : ℤ :=
  a 0 * (b 1 * c 2 - b 2 * c 1) -
  a 1 * (b 0 * c 2 - b 2 * c 0) +
  a 2 * (b 0 * c 1 - b 1 * c 0)

theorem vectors_not_coplanar :
  scalar_triple_product vector_a vector_b vector_c ≠ 0 :=
by
  sorry

end vectors_not_coplanar_l190_190194


namespace right_triangle_hypotenuse_l190_190146

theorem right_triangle_hypotenuse (A : ℝ) (h height : ℝ) :
  A = 320 ∧ height = 16 →
  ∃ c : ℝ, c = 4 * Real.sqrt 116 :=
by
  intro h
  sorry

end right_triangle_hypotenuse_l190_190146


namespace part1_part2_part3_l190_190760

variable (a b c d S A B C D : ℝ)

-- The given conditions
def cond1 : Prop := a + c = b + d
def cond2 : Prop := A + C = B + D
def cond3 : Prop := S^2 = a * b * c * d

-- The statements to prove
theorem part1 (h1 : cond1 a b c d) (h2 : cond2 A B C D) : cond3 a b c d S := sorry
theorem part2 (h1 : cond1 a b c d) (h3 : cond3 a b c d S) : cond2 A B C D := sorry
theorem part3 (h2 : cond2 A B C D) : cond3 a b c d S := sorry

end part1_part2_part3_l190_190760


namespace simplify_expression_l190_190888

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) : (a - 2) * (b - 2) = -2 * m := 
by
  sorry

end simplify_expression_l190_190888


namespace determine_values_l190_190456

-- Define the main problem conditions
def A := 1.2
def B := 12

-- The theorem statement capturing the problem conditions and the solution
theorem determine_values (A B : ℝ) (h1 : A + B = 13.2) (h2 : B = 10 * A) : A = 1.2 ∧ B = 12 :=
  sorry

end determine_values_l190_190456


namespace trigonometric_operation_l190_190983

theorem trigonometric_operation :
  let m := Real.cos (Real.pi / 6)
  let n := Real.sin (Real.pi / 6)
  let op (m n : ℝ) := m^2 - m * n - n^2
  op m n = (1 / 2 : ℝ) - (Real.sqrt 3 / 4) :=
by
  sorry

end trigonometric_operation_l190_190983


namespace chromium_atoms_in_compound_l190_190637

-- Definitions of given conditions
def hydrogen_atoms : Nat := 2
def oxygen_atoms : Nat := 4
def compound_molecular_weight : ℝ := 118
def hydrogen_atomic_weight : ℝ := 1
def chromium_atomic_weight : ℝ := 52
def oxygen_atomic_weight : ℝ := 16

-- Problem statement to find the number of Chromium atoms
theorem chromium_atoms_in_compound (hydrogen_atoms : Nat) (oxygen_atoms : Nat) (compound_molecular_weight : ℝ)
    (hydrogen_atomic_weight : ℝ) (chromium_atomic_weight : ℝ) (oxygen_atomic_weight : ℝ) :
  hydrogen_atoms * hydrogen_atomic_weight + 
  oxygen_atoms * oxygen_atomic_weight + 
  chromium_atomic_weight = compound_molecular_weight → 
  chromium_atomic_weight = 52 :=
by
  sorry

end chromium_atoms_in_compound_l190_190637


namespace average_books_collected_per_day_l190_190665

theorem average_books_collected_per_day :
  let n := 7
  let a := 12
  let d := 12
  let S_n := (n * (2 * a + (n - 1) * d)) / 2
  S_n / n = 48 :=
by
  let n := 7
  let a := 12
  let d := 12
  let S_n := (n * (2 * a + (n - 1) * d)) / 2
  show S_n / n = 48
  sorry

end average_books_collected_per_day_l190_190665


namespace abs_sub_eq_five_l190_190871

theorem abs_sub_eq_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 :=
sorry

end abs_sub_eq_five_l190_190871


namespace area_of_square_l190_190192

-- Defining the points A and B as given in the conditions.
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (4, 6)

-- Theorem statement: proving that the area of the square given the endpoints A and B is 12.5.
theorem area_of_square : 
  ∀ (A B : ℝ × ℝ),
  A = (1, 2) → B = (4, 6) → 
  ∃ (area : ℝ), area = 12.5 := 
by
  intros A B hA hB
  sorry

end area_of_square_l190_190192


namespace grace_apples_after_6_weeks_l190_190791

def apples_per_day_bella : ℕ := 6

def days_per_week : ℕ := 7

def fraction_apples_bella_consumes : ℚ := 1/3

def weeks : ℕ := 6

theorem grace_apples_after_6_weeks :
  let apples_per_week_bella := apples_per_day_bella * days_per_week
  let apples_per_week_grace := apples_per_week_bella / fraction_apples_bella_consumes
  let remaining_apples_week := apples_per_week_grace - apples_per_week_bella
  let total_apples := remaining_apples_week * weeks
  total_apples = 504 := by
  sorry

end grace_apples_after_6_weeks_l190_190791


namespace highest_monthly_profit_max_average_profit_l190_190169

noncomputable def profit (x : ℕ) : ℤ :=
if 1 ≤ x ∧ x ≤ 5 then 26 * x - 56
else if 5 < x ∧ x ≤ 12 then 210 - 20 * x
else 0

noncomputable def average_profit (x : ℕ) : ℝ :=
if 1 ≤ x ∧ x ≤ 5 then (13 * ↑x - 43 : ℤ) / ↑x
else if 5 < x ∧ x ≤ 12 then (-10 * ↑x + 200 - 640 / ↑x : ℝ)
else 0

theorem highest_monthly_profit :
  ∃ m p, m = 6 ∧ p = 90 ∧ profit m = p :=
by sorry

theorem max_average_profit (x : ℕ) :
  1 ≤ x ∧ x ≤ 12 →
  average_profit x ≤ 40 ∧ (average_profit 8 = 40 → x = 8) :=
by sorry

end highest_monthly_profit_max_average_profit_l190_190169


namespace cannot_form_complex_pattern_l190_190329

structure GeometricPieces where
  triangles : Nat
  squares : Nat

def possibleToForm (pieces : GeometricPieces) : Bool :=
  sorry -- Since the formation logic is unknown, it is incomplete.

theorem cannot_form_complex_pattern : 
  let pieces := GeometricPieces.mk 8 7
  ¬ possibleToForm pieces = true := 
sorry

end cannot_form_complex_pattern_l190_190329


namespace magnitude_of_z_l190_190272

open Complex

theorem magnitude_of_z (z : ℂ) (h : z^2 + Complex.normSq z = 4 - 7 * Complex.I) : 
  Complex.normSq z = 65 / 8 := 
by
  sorry

end magnitude_of_z_l190_190272


namespace arithmetic_expression_eval_l190_190896

theorem arithmetic_expression_eval :
  ((26.3 * 12 * 20) / 3) + 125 = 2229 :=
sorry

end arithmetic_expression_eval_l190_190896


namespace business_ownership_l190_190987

variable (x : ℝ) (total_value : ℝ)
variable (fraction_sold : ℝ)
variable (sale_amount : ℝ)

-- Conditions
axiom total_value_condition : total_value = 10000
axiom fraction_sold_condition : fraction_sold = 3 / 5
axiom sale_amount_condition : sale_amount = 2000
axiom equation_condition : (fraction_sold * x * total_value = sale_amount)

theorem business_ownership : x = 1 / 3 := by 
  have hv := total_value_condition
  have hf := fraction_sold_condition
  have hs := sale_amount_condition
  have he := equation_condition
  sorry

end business_ownership_l190_190987


namespace biggest_number_l190_190316

noncomputable def Yoongi_collected : ℕ := 4
noncomputable def Jungkook_collected : ℕ := 6 * 3
noncomputable def Yuna_collected : ℕ := 5

theorem biggest_number :
  Jungkook_collected = 18 ∧ Jungkook_collected > Yoongi_collected ∧ Jungkook_collected > Yuna_collected :=
by
  sorry

end biggest_number_l190_190316


namespace product_of_consecutive_integers_is_perfect_square_l190_190920

theorem product_of_consecutive_integers_is_perfect_square (n : ℤ) :
    n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1) ^ 2 :=
sorry

end product_of_consecutive_integers_is_perfect_square_l190_190920


namespace investment2_rate_l190_190804

-- Define the initial conditions
def total_investment : ℝ := 10000
def investment1 : ℝ := 4000
def rate1 : ℝ := 0.05
def investment2 : ℝ := 3500
def income1 : ℝ := investment1 * rate1
def yearly_income_goal : ℝ := 500
def remaining_investment : ℝ := total_investment - investment1 - investment2
def rate3 : ℝ := 0.064
def income3 : ℝ := remaining_investment * rate3

-- The main theorem
theorem investment2_rate (rate2 : ℝ) : 
  income1 + income3 + investment2 * (rate2 / 100) = yearly_income_goal → rate2 = 4 := 
by 
  sorry

end investment2_rate_l190_190804


namespace degrees_of_remainder_division_l190_190882

theorem degrees_of_remainder_division (f g : Polynomial ℝ) (h : g = Polynomial.C 3 * Polynomial.X ^ 3 + Polynomial.C (-4) * Polynomial.X ^ 2 + Polynomial.C 1 * Polynomial.X + Polynomial.C (-8)) :
  ∃ r q : Polynomial ℝ, f = g * q + r ∧ (r.degree < 3) := 
sorry

end degrees_of_remainder_division_l190_190882


namespace overtime_percentage_increase_l190_190739

-- Define basic conditions
def basic_hours := 40
def total_hours := 48
def basic_pay := 20
def total_wage := 25

-- Calculate overtime hours and wages
def overtime_hours := total_hours - basic_hours
def overtime_pay := total_wage - basic_pay

-- Define basic and overtime hourly rates
def basic_hourly_rate := basic_pay / basic_hours
def overtime_hourly_rate := overtime_pay / overtime_hours

-- Calculate and state the theorem for percentage increase
def percentage_increase := ((overtime_hourly_rate - basic_hourly_rate) / basic_hourly_rate) * 100

theorem overtime_percentage_increase :
  percentage_increase = 25 :=
by
  sorry

end overtime_percentage_increase_l190_190739


namespace distance_with_father_l190_190070

variable (total_distance driven_with_mother driven_with_father: ℝ)

theorem distance_with_father :
  total_distance = 0.67 ∧ driven_with_mother = 0.17 → driven_with_father = 0.50 := 
by
  sorry

end distance_with_father_l190_190070


namespace cone_surface_area_and_volume_l190_190778

theorem cone_surface_area_and_volume
  (r l m : ℝ)
  (h_ratio : (π * r * l) / (π * r * l + π * r^2) = 25 / 32)
  (h_height : m = 96) :
  (π * r * l + π * r^2 = 3584 * π) ∧ ((1 / 3) * π * r^2 * m = 25088 * π) :=
by {
  sorry
}

end cone_surface_area_and_volume_l190_190778


namespace ship_B_has_highest_rt_no_cars_l190_190059

def ship_percentage_with_no_cars (total_rt: ℕ) (percent_with_cars: ℕ) : ℕ :=
  total_rt - (percent_with_cars * total_rt) / 100

theorem ship_B_has_highest_rt_no_cars :
  let A_rt := 30
  let A_with_cars := 25
  let B_rt := 50
  let B_with_cars := 15
  let C_rt := 20
  let C_with_cars := 35
  let A_no_cars := ship_percentage_with_no_cars A_rt A_with_cars
  let B_no_cars := ship_percentage_with_no_cars B_rt B_with_cars
  let C_no_cars := ship_percentage_with_no_cars C_rt C_with_cars
  A_no_cars < B_no_cars ∧ C_no_cars < B_no_cars := by
  sorry

end ship_B_has_highest_rt_no_cars_l190_190059


namespace units_digit_17_pow_27_l190_190586

-- Define the problem: the units digit of 17^27
theorem units_digit_17_pow_27 : (17 ^ 27) % 10 = 3 :=
sorry

end units_digit_17_pow_27_l190_190586


namespace find_whole_number_M_l190_190721

theorem find_whole_number_M (M : ℕ) (h : 8 < M / 4 ∧ M / 4 < 9) : M = 33 :=
sorry

end find_whole_number_M_l190_190721


namespace jill_speed_is_8_l190_190423

-- Definitions for conditions
def speed_jack1 := 12 -- speed in km/h for the first 12 km
def distance_jack1 := 12 -- distance in km for the first 12 km

def speed_jack2 := 6 -- speed in km/h for the second 12 km
def distance_jack2 := 12 -- distance in km for the second 12 km

def distance_jill := distance_jack1 + distance_jack2 -- total distance in km for Jill

-- Total time taken by Jack
def time_jack := (distance_jack1 / speed_jack1) + (distance_jack2 / speed_jack2)

-- Jill's speed calculation
def jill_speed := distance_jill / time_jack

-- Theorem stating Jill's speed is 8 km/h
theorem jill_speed_is_8 : jill_speed = 8 := by
  sorry

end jill_speed_is_8_l190_190423


namespace belize_homes_l190_190749

theorem belize_homes (H : ℝ) 
  (h1 : (3 / 5) * (3 / 4) * H = 240) : 
  H = 400 :=
sorry

end belize_homes_l190_190749


namespace cost_per_gallon_is_45_l190_190452

variable (totalArea coverage cost_jason cost_jeremy dollars_per_gallon : ℕ)

-- Conditions
def total_area := 1600
def coverage_per_gallon := 400
def num_coats := 2
def contribution_jason := 180
def contribution_jeremy := 180

-- Gallons needed calculation
def gallons_per_coat := total_area / coverage_per_gallon
def total_gallons := gallons_per_coat * num_coats

-- Total cost calculation
def total_cost := contribution_jason + contribution_jeremy

-- Cost per gallon calculation
def cost_per_gallon := total_cost / total_gallons

-- Proof statement
theorem cost_per_gallon_is_45 : cost_per_gallon = 45 :=
by
  sorry

end cost_per_gallon_is_45_l190_190452


namespace log_relationship_l190_190306

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem log_relationship :
  a > b ∧ b > c := by
  sorry

end log_relationship_l190_190306


namespace unique_base_for_final_digit_one_l190_190012

theorem unique_base_for_final_digit_one :
  ∃! b : ℕ, 2 ≤ b ∧ b ≤ 15 ∧ 648 % b = 1 :=
by {
  sorry
}

end unique_base_for_final_digit_one_l190_190012


namespace pyramid_area_ratio_l190_190096

theorem pyramid_area_ratio (S S1 S2 : ℝ) (h1 : S1 = (99 / 100)^2 * S) (h2 : S2 = (1 / 100)^2 * S) :
  S1 / S2 = 9801 := by
  sorry

end pyramid_area_ratio_l190_190096


namespace side_length_of_S2_l190_190210

variables (s r : ℕ)

-- Conditions
def combined_width_eq : Prop := 3 * s + 100 = 4000
def combined_height_eq : Prop := 2 * r + s = 2500

-- Conclusion we want to prove
theorem side_length_of_S2 : combined_width_eq s → combined_height_eq s r → s = 1300 :=
by
  intros h_width h_height
  sorry

end side_length_of_S2_l190_190210


namespace musketeer_statements_triplets_count_l190_190689

-- Definitions based on the conditions
def musketeers : Type := { x : ℕ // x < 3 }

def is_guilty (m : musketeers) : Prop := sorry  -- Placeholder for the property of being guilty

def statement (m1 m2 : musketeers) : Prop := sorry  -- Placeholder for the statement made by one musketeer about another

-- Condition that each musketeer makes one statement
def made_statement (m : musketeers) : Prop := sorry

-- Condition that exactly one musketeer lied
def exactly_one_lied : Prop := sorry

-- The final proof problem statement:
theorem musketeer_statements_triplets_count : ∃ n : ℕ, n = 99 :=
  sorry

end musketeer_statements_triplets_count_l190_190689


namespace sum_distances_saham_and_mother_l190_190516

theorem sum_distances_saham_and_mother :
  let saham_distance := 2.6
  let mother_distance := 5.98
  saham_distance + mother_distance = 8.58 :=
by
  sorry

end sum_distances_saham_and_mother_l190_190516


namespace part_I_part_II_l190_190170

def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem part_I (x : ℝ) : (f x > 4) ↔ (x < -1.5 ∨ x > 2.5) :=
by
  sorry

theorem part_II (x : ℝ) : ∀ x : ℝ, f x ≥ 3 :=
by
  sorry

end part_I_part_II_l190_190170


namespace new_marketing_percentage_l190_190469

theorem new_marketing_percentage 
  (total_students : ℕ)
  (initial_finance_percentage : ℕ)
  (initial_marketing_percentage : ℕ)
  (initial_operations_management_percentage : ℕ)
  (new_finance_percentage : ℕ)
  (operations_management_percentage : ℕ)
  (total_percentage : ℕ) :
  total_students = 5000 →
  initial_finance_percentage = 85 →
  initial_marketing_percentage = 80 →
  initial_operations_management_percentage = 10 →
  new_finance_percentage = 92 →
  operations_management_percentage = 10 →
  total_percentage = 175 →
  initial_marketing_percentage - (new_finance_percentage - initial_finance_percentage) = 73 :=
by
  sorry

end new_marketing_percentage_l190_190469


namespace find_sum_of_xyz_l190_190857

theorem find_sum_of_xyz : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  (151 / 44 : ℚ) = 3 + 1 / (x + 1 / (y + 1 / z)) ∧ x + y + z = 11 :=
by 
  sorry

end find_sum_of_xyz_l190_190857


namespace jerry_charge_per_hour_l190_190391

-- Define the conditions from the problem
def time_painting : ℝ := 8
def time_fixing_counter : ℝ := 3 * time_painting
def time_mowing_lawn : ℝ := 6
def total_time_worked : ℝ := time_painting + time_fixing_counter + time_mowing_lawn
def total_payment : ℝ := 570

-- The proof statement
theorem jerry_charge_per_hour : 
  total_payment / total_time_worked = 15 :=
by
  sorry

end jerry_charge_per_hour_l190_190391


namespace remainder_when_xy_div_by_22_l190_190986

theorem remainder_when_xy_div_by_22
  (x y : ℤ)
  (h1 : x % 126 = 37)
  (h2 : y % 176 = 46) : 
  (x + y) % 22 = 21 := by
  sorry

end remainder_when_xy_div_by_22_l190_190986


namespace total_candies_in_third_set_l190_190048

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l190_190048


namespace longest_collection_pages_l190_190855

theorem longest_collection_pages 
    (pages_per_inch_miles : ℕ := 5) 
    (pages_per_inch_daphne : ℕ := 50) 
    (height_miles : ℕ := 240) 
    (height_daphne : ℕ := 25) : 
  max (pages_per_inch_miles * height_miles) (pages_per_inch_daphne * height_daphne) = 1250 := 
by
  sorry

end longest_collection_pages_l190_190855


namespace simplify_fraction_l190_190045

theorem simplify_fraction :
  (1 / (1 / (Real.sqrt 3 + 1) + 3 / (Real.sqrt 5 - 2))) = 2 / (Real.sqrt 3 + 6 * Real.sqrt 5 + 11) :=
by
  sorry

end simplify_fraction_l190_190045


namespace smallest_yellow_marbles_l190_190352

theorem smallest_yellow_marbles :
  ∃ n : ℕ, (n ≡ 0 [MOD 20]) ∧
           (∃ b : ℕ, b = n / 4) ∧
           (∃ r : ℕ, r = n / 5) ∧
           (∃ g : ℕ, g = 10) ∧
           (∃ y : ℕ, y = n - (b + r + g) ∧ y = 1) :=
sorry

end smallest_yellow_marbles_l190_190352


namespace center_of_symmetry_is_neg2_3_l190_190782

theorem center_of_symmetry_is_neg2_3 :
  ∃ (a b : ℝ), 
  (a,b) = (-2, 3) ∧ 
  ∀ x : ℝ, 
    2 * b = ((a + x + 2)^3 - (a + x) + 1) + ((a - x + 2)^3 - (a - x) + 1) := 
by
  use -2, 3
  sorry

end center_of_symmetry_is_neg2_3_l190_190782


namespace nat_know_albums_l190_190260

/-- Define the number of novels, comics, documentaries and crates properties --/
def novels := 145
def comics := 271
def documentaries := 419
def crates := 116
def items_per_crate := 9

/-- Define the total capacity of crates --/
def total_capacity := crates * items_per_crate

/-- Define the total number of other items --/
def other_items := novels + comics + documentaries

/-- Define the number of albums --/
def albums := total_capacity - other_items

/-- Theorem: Prove that the number of albums is equal to 209 --/
theorem nat_know_albums : albums = 209 := by
  sorry

end nat_know_albums_l190_190260


namespace dividend_from_tonys_stock_l190_190982

theorem dividend_from_tonys_stock (investment price_per_share total_income : ℝ) 
  (h1 : investment = 3200) (h2 : price_per_share = 85) (h3 : total_income = 250) : 
  (total_income / (investment / price_per_share)) = 6.76 :=
by 
  sorry

end dividend_from_tonys_stock_l190_190982


namespace binary_to_decimal_correct_l190_190837

def binary_to_decimal : ℕ := 110011

theorem binary_to_decimal_correct : 
  binary_to_decimal = 51 := sorry

end binary_to_decimal_correct_l190_190837


namespace bus_ride_cost_l190_190196

theorem bus_ride_cost (B T : ℝ) 
  (h1 : T = B + 6.85)
  (h2 : T + B = 9.65)
  (h3 : ∃ n : ℤ, B = 0.35 * n ∧ ∃ m : ℤ, T = 0.35 * m) : 
  B = 1.40 := 
by
  sorry

end bus_ride_cost_l190_190196


namespace team_selection_count_l190_190204

-- The problem's known conditions
def boys : ℕ := 10
def girls : ℕ := 12
def team_size : ℕ := 8

-- The number of ways to select a team of 8 members with at least 2 boys and no more than 4 boys
noncomputable def count_ways : ℕ :=
  (Nat.choose boys 2) * (Nat.choose girls 6) +
  (Nat.choose boys 3) * (Nat.choose girls 5) +
  (Nat.choose boys 4) * (Nat.choose girls 4)

-- The main statement to prove
theorem team_selection_count : count_ways = 238570 := by
  sorry

end team_selection_count_l190_190204


namespace triangle_side_relation_l190_190438

theorem triangle_side_relation (a b c : ℝ) 
    (h_angles : 55 = 55 ∧ 15 = 15 ∧ 110 = 110) :
    c^2 - a^2 = a * b :=
  sorry

end triangle_side_relation_l190_190438


namespace compute_expression_l190_190470

theorem compute_expression : 9 + 7 * (5 - Real.sqrt 16)^2 = 16 := by
  sorry

end compute_expression_l190_190470


namespace no_combination_of_three_coins_sums_to_52_cents_l190_190458

def is_valid_coin (c : ℕ) : Prop :=
  c = 5 ∨ c = 10 ∨ c = 25 ∨ c = 50 ∨ c = 100

theorem no_combination_of_three_coins_sums_to_52_cents :
  ¬ ∃ a b c : ℕ, is_valid_coin a ∧ is_valid_coin b ∧ is_valid_coin c ∧ a + b + c = 52 :=
by 
  sorry

end no_combination_of_three_coins_sums_to_52_cents_l190_190458


namespace factorial_trailing_zeros_500_l190_190548

theorem factorial_trailing_zeros_500 :
  let count_factors_of_five (n : ℕ) : ℕ := n / 5 + n / 25 + n / 125
  count_factors_of_five 500 = 124 :=
by
  sorry  -- The proof is not required as per the instructions.

end factorial_trailing_zeros_500_l190_190548


namespace probability_of_exactly_one_success_probability_of_at_least_one_success_l190_190694

variable (PA : ℚ := 1/2)
variable (PB : ℚ := 2/5)
variable (P_A_bar : ℚ := 1 - PA)
variable (P_B_bar : ℚ := 1 - PB)

theorem probability_of_exactly_one_success :
  PA * P_B_bar + PB * P_A_bar = 1/2 :=
sorry

theorem probability_of_at_least_one_success :
  1 - (P_A_bar * P_A_bar * P_B_bar * P_B_bar) = 91/100 :=
sorry

end probability_of_exactly_one_success_probability_of_at_least_one_success_l190_190694


namespace sale_price_relationship_l190_190285

/-- Elaine's Gift Shop increased the original prices of all items by 10% 
  and then offered a 30% discount on these new prices in a clearance sale 
  - proving the relationship between the final sale price and the original price of an item -/

theorem sale_price_relationship (p : ℝ) : 
  (0.7 * (1.1 * p) = 0.77 * p) :=
by 
  sorry

end sale_price_relationship_l190_190285


namespace boys_from_clay_l190_190216

theorem boys_from_clay (total_students jonas_students clay_students pine_students total_boys total_girls : ℕ)
  (jonas_girls pine_boys : ℕ) 
  (H1 : total_students = 150)
  (H2 : jonas_students = 50)
  (H3 : clay_students = 60)
  (H4 : pine_students = 40)
  (H5 : total_boys = 80)
  (H6 : total_girls = 70)
  (H7 : jonas_girls = 30)
  (H8 : pine_boys = 15):
  ∃ (clay_boys : ℕ), clay_boys = 45 :=
by
  have jonas_boys : ℕ := jonas_students - jonas_girls
  have boys_from_clay := total_boys - pine_boys - jonas_boys
  exact ⟨boys_from_clay, by sorry⟩

end boys_from_clay_l190_190216


namespace lateral_surface_area_of_parallelepiped_is_correct_l190_190603

noncomputable def lateral_surface_area (diagonal : ℝ) (angle : ℝ) (base_area : ℝ) : ℝ :=
  let h := diagonal * Real.sin angle
  let s := diagonal * Real.cos angle
  let side1_sq := s ^ 2  -- represents DC^2 + AD^2
  let base_diag_sq := 25  -- already given as 25 from BD^2
  let added := side1_sq + 2 * base_area
  2 * h * Real.sqrt added

theorem lateral_surface_area_of_parallelepiped_is_correct :
  lateral_surface_area 10 (Real.pi / 3) 12 = 70 * Real.sqrt 3 :=
by
  sorry

end lateral_surface_area_of_parallelepiped_is_correct_l190_190603


namespace part_a_part_b_l190_190746

/-- Definition of the sequence of numbers on the cards -/
def card_numbers (n : ℕ) : ℕ :=
  if n = 0 then 1 else (10^(n + 1) - 1) / 9 * 2 + 1

/-- Part (a) statement: Is it possible to choose at least three cards such that 
the sum of the numbers on them equals a number where all digits except one are twos? -/
theorem part_a : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ card_numbers a + card_numbers b + card_numbers c % 10 = 2 ∧ 
  (∀ d, ∃ k ≤ 1, (card_numbers a + card_numbers b + card_numbers c / (10^d)) % 10 = 2) :=
sorry

/-- Part (b) statement: Suppose several cards were chosen such that the sum of the numbers 
on them equals a number where all digits except one are twos. What could be the digit that is not two? -/
theorem part_b (sum : ℕ) :
  (∀ d, sum / (10^d) % 10 = 2) → ((sum % 10 = 0) ∨ (sum % 10 = 1)) :=
sorry

end part_a_part_b_l190_190746


namespace set_representation_l190_190537

def is_Natural (n : ℕ) : Prop :=
  n ≠ 0

def condition (x : ℕ) : Prop :=
  x^2 - 3*x < 0

theorem set_representation :
  {x : ℕ | condition x ∧ is_Natural x} = {1, 2} := 
sorry

end set_representation_l190_190537


namespace max_value_of_ab_l190_190003

theorem max_value_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 5 * a + 3 * b < 90) :
  ab * (90 - 5 * a - 3 * b) ≤ 1800 :=
sorry

end max_value_of_ab_l190_190003


namespace simplify_expression_l190_190560

theorem simplify_expression (a b : ℝ) : a + (5 * a - 3 * b) = 6 * a - 3 * b :=
by sorry

end simplify_expression_l190_190560


namespace basic_computer_price_l190_190693

theorem basic_computer_price (C P : ℝ) 
(h1 : C + P = 2500) 
(h2 : P = (1 / 6) * (C + 500 + P)) : 
  C = 2000 :=
by
  sorry

end basic_computer_price_l190_190693


namespace polygon_angle_multiple_l190_190639

theorem polygon_angle_multiple (m : ℕ) (h : m ≥ 3) : 
  (∃ k : ℕ, (2 * m - 2) * 180 = k * ((m - 2) * 180)) ↔ (m = 3 ∨ m = 4) :=
by sorry

end polygon_angle_multiple_l190_190639


namespace distinct_nonzero_reals_equation_l190_190011

theorem distinct_nonzero_reals_equation {a b c d : ℝ} 
  (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0) 
  (h₄ : a ≠ b) (h₅ : b ≠ c) (h₆ : c ≠ d) (h₇ : d ≠ a) (h₈ : a ≠ c) (h₉ : b ≠ d)
  (h₁₀ : a * c = b * d) 
  (h₁₁ : a / b + b / c + c / d + d / a = 4) :
  (a / c + c / a + b / d + d / b = 4) :=
by
  sorry

end distinct_nonzero_reals_equation_l190_190011


namespace determine_a_l190_190032

theorem determine_a (a : ℝ) : (∃ b : ℝ, (3 * (x : ℝ))^2 - 2 * 3 * b * x + b^2 = 9 * x^2 - 27 * x + a) → a = 20.25 :=
by
  sorry

end determine_a_l190_190032


namespace sum_of_cubes_eq_zero_l190_190288

theorem sum_of_cubes_eq_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -4) : a^3 + b^3 = 0 :=
sorry

end sum_of_cubes_eq_zero_l190_190288


namespace bianca_marathon_total_miles_l190_190539

theorem bianca_marathon_total_miles : 8 + 4 = 12 :=
by
  sorry

end bianca_marathon_total_miles_l190_190539


namespace sin_X_value_l190_190166

variables (a b X : ℝ)

-- Conditions
def conditions :=
  (1/2 * a * b * Real.sin X = 100) ∧ (Real.sqrt (a * b) = 15)

theorem sin_X_value (h : conditions a b X) : Real.sin X = 8 / 9 := by
  sorry

end sin_X_value_l190_190166


namespace sandy_marks_loss_l190_190630

theorem sandy_marks_loss (n m c p : ℕ) (h1 : n = 30) (h2 : m = 65) (h3 : c = 25) (h4 : p = 3) :
  ∃ x : ℕ, (c * p - m) / (n - c) = x ∧ x = 2 := by
  sorry

end sandy_marks_loss_l190_190630


namespace savings_by_paying_cash_l190_190049

theorem savings_by_paying_cash
  (cash_price : ℕ) (down_payment : ℕ) (monthly_payment : ℕ) (number_of_months : ℕ)
  (h1 : cash_price = 400) (h2 : down_payment = 120) (h3 : monthly_payment = 30) (h4 : number_of_months = 12) :
  cash_price + (monthly_payment * number_of_months - down_payment) - cash_price = 80 :=
by
  sorry

end savings_by_paying_cash_l190_190049


namespace tyrone_gave_15_marbles_l190_190995

variables (x : ℕ)

-- Define initial conditions for Tyrone and Eric
def initial_tyrone := 120
def initial_eric := 20

-- Define the condition after giving marbles
def condition_after_giving (x : ℕ) := 120 - x = 3 * (20 + x)

theorem tyrone_gave_15_marbles (x : ℕ) : condition_after_giving x → x = 15 :=
by
  intro h
  sorry

end tyrone_gave_15_marbles_l190_190995


namespace polynomial_calculation_l190_190389

theorem polynomial_calculation :
  (49^5 - 5 * 49^4 + 10 * 49^3 - 10 * 49^2 + 5 * 49 - 1) = 254804368 :=
by
  sorry

end polynomial_calculation_l190_190389


namespace largest_house_number_l190_190575

theorem largest_house_number (phone_number_digits : List ℕ) (house_number_digits : List ℕ) :
  phone_number_digits = [5, 0, 4, 9, 3, 2, 6] →
  phone_number_digits.sum = 29 →
  (∀ (d1 d2 : ℕ), d1 ∈ house_number_digits → d2 ∈ house_number_digits → d1 ≠ d2) →
  house_number_digits.sum = 29 →
  house_number_digits = [9, 8, 7, 5] :=
by
  intros
  sorry

end largest_house_number_l190_190575


namespace number_division_reduction_l190_190596

theorem number_division_reduction (x : ℕ) (h : x / 3 = x - 24) : x = 36 := sorry

end number_division_reduction_l190_190596


namespace range_of_a_l190_190810

theorem range_of_a
  (a : ℝ)
  (h1 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 ≥ a)
  (h2 : ∃ x0 : ℝ, x0^2 + 2*a*x0 + 2 - a = 0) :
  a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l190_190810


namespace intersection_correct_l190_190261

open Set

def M : Set ℤ := {0, 1, 2, -1}
def N : Set ℤ := {0, 1, 2, 3}

theorem intersection_correct : M ∩ N = {0, 1, 2} :=
by 
  -- Proof omitted
  sorry

end intersection_correct_l190_190261


namespace danny_distance_to_work_l190_190480

-- Define the conditions and the problem in terms of Lean definitions
def distance_to_first_friend : ℕ := 8
def distance_to_second_friend : ℕ := distance_to_first_friend / 2
def total_distance_driven_so_far : ℕ := distance_to_first_friend + distance_to_second_friend
def distance_to_work : ℕ := 3 * total_distance_driven_so_far

-- Lean statement to be proven
theorem danny_distance_to_work :
  distance_to_work = 36 :=
by
  -- This is the proof placeholder
  sorry

end danny_distance_to_work_l190_190480


namespace find_m_n_l190_190031

theorem find_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (hmn : m^n = n^(m - n)) : 
  (m = 9 ∧ n = 3) ∨ (m = 8 ∧ n = 2) :=
sorry

end find_m_n_l190_190031


namespace pears_to_peaches_l190_190592

-- Define the weights of pears and peaches
variables (pear peach : ℝ) 

-- Given conditions: 9 pears weigh the same as 6 peaches
axiom weight_ratio : 9 * pear = 6 * peach

-- Theorem to prove: 36 pears weigh the same as 24 peaches
theorem pears_to_peaches (h : 9 * pear = 6 * peach) : 36 * pear = 24 * peach :=
by
  sorry

end pears_to_peaches_l190_190592


namespace sq_97_l190_190686

theorem sq_97 : 97^2 = 9409 :=
by
  sorry

end sq_97_l190_190686


namespace problem_a_problem_b_problem_c_problem_d_l190_190564

variable {a b : ℝ}

theorem problem_a (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) : ab ≤ 1 / 8 := sorry

theorem problem_b (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  (1 / a) + (8 / b) ≥ 25 := sorry

theorem problem_c (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  a^2 + 4 * b^2 ≥ 1 / 2 := sorry

theorem problem_d (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  a^2 - b^2 > -1 / 4 := sorry

end problem_a_problem_b_problem_c_problem_d_l190_190564


namespace unique_zero_location_l190_190303

theorem unique_zero_location (f : ℝ → ℝ) (h : ∃! x, f x = 0 ∧ 1 < x ∧ x < 3) :
  ¬ (∃ x, 2 < x ∧ x < 5 ∧ f x = 0) :=
sorry

end unique_zero_location_l190_190303


namespace sum_series_l190_190559

noncomputable def f (n : ℕ) : ℝ :=
  (6 * (n : ℝ)^3 - 3 * (n : ℝ)^2 + 2 * (n : ℝ) - 1) / 
  ((n : ℝ) * ((n : ℝ) - 1) * ((n : ℝ)^2 + (n : ℝ) + 1) * ((n : ℝ)^2 - (n : ℝ) + 1))

theorem sum_series:
  (∑' n, if h : 2 ≤ n then f n else 0) = 1 := 
by
  sorry

end sum_series_l190_190559


namespace range_of_a_l190_190820

/-- 
For the system of inequalities in terms of x 
    \begin{cases} 
    x - a < 0 
    ax < 1 
    \end{cases}
the range of values for the real number a such that the solution set is not empty is [-1, ∞).
-/
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x - a < 0 ∧ a * x < 1) ↔ -1 ≤ a :=
by sorry

end range_of_a_l190_190820


namespace quarters_number_l190_190999

theorem quarters_number (total_value : ℝ)
    (bills1 : ℝ := 2)
    (bill5 : ℝ := 5)
    (dimes : ℝ := 20 * 0.1)
    (nickels : ℝ := 8 * 0.05)
    (pennies : ℝ := 35 * 0.01) :
    total_value = 13 → (total_value - (bills1 + bill5 + dimes + nickels + pennies)) / 0.25 = 13 :=
by
  intro h
  have h_total := h
  sorry

end quarters_number_l190_190999


namespace lambda_sum_ellipse_l190_190998

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

noncomputable def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 4)

noncomputable def intersects_y_axis (k : ℝ) : ℝ × ℝ :=
  (0, -4 * k)

noncomputable def lambda1 (x1 : ℝ) : ℝ :=
  x1 / (4 - x1)

noncomputable def lambda2 (x2 : ℝ) : ℝ :=
  x2 / (4 - x2)

theorem lambda_sum_ellipse {k x1 x2 : ℝ}
  (h1 : ellipse x1 (k * (x1 - 4)))
  (h2 : ellipse x2 (k * (x2 - 4)))
  (h3 : line_through_focus k x1 (k * (x1 - 4)))
  (h4 : line_through_focus k x2 (k * (x2 - 4))) :
  lambda1 x1 + lambda2 x2 = -50 / 9 := 
sorry

end lambda_sum_ellipse_l190_190998


namespace gf_neg3_eq_1262_l190_190702

def f (x : ℤ) : ℤ := x^3 + 6
def g (x : ℤ) : ℤ := 3 * x^2 + 3 * x + 2

theorem gf_neg3_eq_1262 : g (f (-3)) = 1262 := by
  sorry

end gf_neg3_eq_1262_l190_190702


namespace probability_neither_event_l190_190093

-- Definitions of given probabilities
def P_soccer_match : ℚ := 5 / 8
def P_science_test : ℚ := 1 / 4

-- Calculations of the complements
def P_no_soccer_match : ℚ := 1 - P_soccer_match
def P_no_science_test : ℚ := 1 - P_science_test

-- Independence of events implies the probability of neither event is the product of their complements
theorem probability_neither_event :
  (P_no_soccer_match * P_no_science_test) = 9 / 32 :=
by
  sorry

end probability_neither_event_l190_190093


namespace barbara_current_savings_l190_190117

def wristwatch_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def initial_saving_duration : ℕ := 10
def further_saving_duration : ℕ := 16

theorem barbara_current_savings : 
  -- Given:
  -- wristwatch_cost: $100
  -- weekly_allowance: $5
  -- further_saving_duration: 16 weeks
  -- Prove:
  -- Barbara currently has $20
  wristwatch_cost - weekly_allowance * further_saving_duration = 20 :=
by
  sorry

end barbara_current_savings_l190_190117


namespace pizza_shared_cost_l190_190808

theorem pizza_shared_cost (total_price : ℕ) (num_people : ℕ) (share: ℕ)
  (h1 : total_price = 40) (h2 : num_people = 5) : share = 8 :=
by
  sorry

end pizza_shared_cost_l190_190808


namespace Carter_reads_30_pages_in_1_hour_l190_190809

variables (C L O : ℕ)

def Carter_reads_half_as_many_pages_as_Lucy_in_1_hour (C L : ℕ) : Prop :=
  C = L / 2

def Lucy_reads_20_more_pages_than_Oliver_in_1_hour (L O : ℕ) : Prop :=
  L = O + 20

def Oliver_reads_40_pages_in_1_hour (O : ℕ) : Prop :=
  O = 40

theorem Carter_reads_30_pages_in_1_hour
  (C L O : ℕ)
  (h1 : Carter_reads_half_as_many_pages_as_Lucy_in_1_hour C L)
  (h2 : Lucy_reads_20_more_pages_than_Oliver_in_1_hour L O)
  (h3 : Oliver_reads_40_pages_in_1_hour O) : 
  C = 30 :=
by
  sorry

end Carter_reads_30_pages_in_1_hour_l190_190809


namespace cos_330_is_sqrt3_over_2_l190_190006

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l190_190006


namespace octagon_area_in_square_l190_190525

/--
An octagon is inscribed in a square such that each vertex of the octagon cuts off a corner
triangle from the square. Each triangle has legs equal to one-fourth of the square's side.
If the perimeter of the square is 160 centimeters, what is the area of the octagon?
-/
theorem octagon_area_in_square
  (side_of_square : ℝ)
  (h1 : 4 * (side_of_square / 4) = side_of_square)
  (h2 : 8 * (side_of_square / 4) = side_of_square)
  (perimeter_of_square : ℝ)
  (h3 : perimeter_of_square = 160)
  (area_of_square : ℝ)
  (h4 : area_of_square = side_of_square^2)
  : ∃ (area_of_octagon : ℝ), area_of_octagon = 1400 := by
  sorry

end octagon_area_in_square_l190_190525


namespace negate_proposition_l190_190434

open Classical

variable (x : ℝ)

theorem negate_proposition :
  (¬ ∀ x : ℝ, x^2 + 2 * x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0 :=
by
  sorry

end negate_proposition_l190_190434


namespace bug_total_distance_l190_190237

/-!
# Problem Statement
A bug starts crawling on a number line from position -3. It first moves to -7, then turns around and stops briefly at 0 before continuing on to 8. Prove that the total distance the bug crawls is 19 units.
-/

def bug_initial_position : ℤ := -3
def bug_position_1 : ℤ := -7
def bug_position_2 : ℤ := 0
def bug_final_position : ℤ := 8

theorem bug_total_distance : 
  |bug_position_1 - bug_initial_position| + 
  |bug_position_2 - bug_position_1| + 
  |bug_final_position - bug_position_2| = 19 :=
by 
  sorry

end bug_total_distance_l190_190237


namespace monomial_2023_eq_l190_190844

def monomial (n : ℕ) : ℤ × ℕ :=
  ((-1)^(n+1) * (2*n - 1), n)

theorem monomial_2023_eq : monomial 2023 = (4045, 2023) :=
by
  sorry

end monomial_2023_eq_l190_190844


namespace problem_statement_l190_190793

noncomputable def floor_T (u v w x : ℝ) : ℤ :=
  ⌊u + v + w + x⌋

theorem problem_statement (u v w x : ℝ) (T : ℝ) (h₁: u^2 + v^2 = 3005) (h₂: w^2 + x^2 = 3005) (h₃: u * w = 1729) (h₄: v * x = 1729) :
  floor_T u v w x = 155 :=
by
  sorry

end problem_statement_l190_190793


namespace tessa_initial_apples_l190_190474

-- Define conditions as variables
variable (initial_apples anita_gave : ℕ)
variable (apples_needed_for_pie : ℕ := 10)
variable (apples_additional_now_needed : ℕ := 1)

-- Define the current amount of apples Tessa has
noncomputable def current_apples :=
  apples_needed_for_pie - apples_additional_now_needed

-- Define the initial apples Tessa had before Anita gave her 5 apples
noncomputable def initial_apples_calculated :=
  current_apples - anita_gave

-- Lean statement to prove the initial number of apples Tessa had
theorem tessa_initial_apples (h_initial_apples : anita_gave = 5) : initial_apples_calculated = 4 :=
by
  -- Here is where the proof would go; we use sorry to indicate it's not provided
  sorry

end tessa_initial_apples_l190_190474


namespace problem1_problem2_problem3_problem4_l190_190105

-- Problem 1: Prove X = 93 given X - 12 = 81
theorem problem1 (X : ℝ) (h : X - 12 = 81) : X = 93 :=
by
  sorry

-- Problem 2: Prove X = 5.4 given 5.1 + X = 10.5
theorem problem2 (X : ℝ) (h : 5.1 + X = 10.5) : X = 5.4 :=
by
  sorry

-- Problem 3: Prove X = 0.7 given 6X = 4.2
theorem problem3 (X : ℝ) (h : 6 * X = 4.2) : X = 0.7 :=
by
  sorry

-- Problem 4: Prove X = 5 given X ÷ 0.4 = 12.5
theorem problem4 (X : ℝ) (h : X / 0.4 = 12.5) : X = 5 :=
by
  sorry

end problem1_problem2_problem3_problem4_l190_190105


namespace divisibility_equivalence_l190_190159

theorem divisibility_equivalence (a b c d : ℤ) (h : a ≠ c) :
  (a - c) ∣ (a * b + c * d) ↔ (a - c) ∣ (a * d + b * c) :=
by
  sorry

end divisibility_equivalence_l190_190159


namespace number_of_terms_in_expansion_l190_190047

theorem number_of_terms_in_expansion :
  (∃ (a1 a2 a3 a4 a5 b1 b2 b3 b4 c1 c2 c3 : ℕ), (a1 + a2 + a3 + a4 + a5) * (b1 + b2 + b3 + b4) * (c1 + c2 + c3) = 60) :=
by
  sorry

end number_of_terms_in_expansion_l190_190047


namespace attendees_not_from_A_B_C_D_l190_190141

theorem attendees_not_from_A_B_C_D
  (num_A : ℕ) (num_B : ℕ) (num_C : ℕ) (num_D : ℕ) (total_attendees : ℕ)
  (hA : num_A = 30)
  (hB : num_B = 2 * num_A)
  (hC : num_C = num_A + 10)
  (hD : num_D = num_C - 5)
  (hTotal : total_attendees = 185)
  : total_attendees - (num_A + num_B + num_C + num_D) = 20 := by
  sorry

end attendees_not_from_A_B_C_D_l190_190141


namespace no_real_solution_for_quadratic_eq_l190_190484

theorem no_real_solution_for_quadratic_eq (y : ℝ) :
  (8 * y^2 + 155 * y + 3) / (4 * y + 45) = 4 * y + 3 →  (¬ ∃ y : ℝ, (8 * y^2 + 37 * y + 33/2 = 0)) :=
by
  sorry

end no_real_solution_for_quadratic_eq_l190_190484


namespace archer_probability_less_than_8_l190_190815

-- Define the conditions as probabilities for hitting the 10-ring, 9-ring, and 8-ring.
def p_10 : ℝ := 0.24
def p_9 : ℝ := 0.28
def p_8 : ℝ := 0.19

-- Define the probability that the archer scores at least 8.
def p_at_least_8 : ℝ := p_10 + p_9 + p_8

-- Calculate the probability of the archer scoring less than 8.
def p_less_than_8 : ℝ := 1 - p_at_least_8

-- Now, state the theorem to prove that this probability is equal to 0.29.
theorem archer_probability_less_than_8 : p_less_than_8 = 0.29 := by sorry

end archer_probability_less_than_8_l190_190815


namespace steve_berry_picking_strategy_l190_190684

def berry_picking_goal_reached (monday_earnings tuesday_earnings total_goal: ℕ) : Prop :=
  monday_earnings + tuesday_earnings >= total_goal

def optimal_thursday_strategy (remaining_goal payment_per_pound total_capacity : ℕ) : ℕ :=
  if remaining_goal = 0 then 0 else total_capacity

theorem steve_berry_picking_strategy :
  let monday_lingonberries := 8
  let monday_cloudberries := 10
  let monday_blueberries := 30 - monday_lingonberries - monday_cloudberries
  let tuesday_lingonberries := 3 * monday_lingonberries
  let tuesday_cloudberries := 2 * monday_cloudberries
  let tuesday_blueberries := 5
  let lingonberry_rate := 2
  let cloudberry_rate := 3
  let blueberry_rate := 5
  let max_capacity := 30
  let total_goal := 150

  let monday_earnings := (monday_lingonberries * lingonberry_rate) + 
                         (monday_cloudberries * cloudberry_rate) + 
                         (monday_blueberries * blueberry_rate)
                         
  let tuesday_earnings := (tuesday_lingonberries * lingonberry_rate) + 
                          (tuesday_cloudberries * cloudberry_rate) +
                          (tuesday_blueberries * blueberry_rate)

  let total_earnings := monday_earnings + tuesday_earnings

  berry_picking_goal_reached monday_earnings tuesday_earnings total_goal ∧
  optimal_thursday_strategy (total_goal - total_earnings) blueberry_rate max_capacity = 30 
:= by {
  sorry
}

end steve_berry_picking_strategy_l190_190684


namespace combined_resistance_parallel_l190_190579

theorem combined_resistance_parallel (x y r : ℝ) (hx : x = 4) (hy : y = 5)
  (h_combined : 1 / r = 1 / x + 1 / y) : r = 20 / 9 := by
  sorry

end combined_resistance_parallel_l190_190579


namespace constants_sum_l190_190954

theorem constants_sum (A B C D : ℕ) 
  (h : ∀ n : ℕ, n ≥ 4 → n^4 = A * (n.choose 4) + B * (n.choose 3) + C * (n.choose 2) + D * (n.choose 1)) 
  : A + B + C + D = 75 :=
by
  sorry

end constants_sum_l190_190954


namespace matrix_power_A_100_l190_190770

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![0, 0, 1],![1, 0, 0],![0, 1, 0]]

theorem matrix_power_A_100 : A^100 = A := by sorry

end matrix_power_A_100_l190_190770


namespace sin_range_l190_190848

theorem sin_range (x : ℝ) (h : x ∈ Set.Icc (Real.pi / 6) (Real.pi / 2)) : 
  Set.range (fun x => Real.sin x) = Set.Icc (1/2 : ℝ) 1 :=
sorry

end sin_range_l190_190848


namespace randy_biscuits_l190_190439

theorem randy_biscuits (initial_biscuits father_gift mother_gift brother_ate : ℕ) : 
  (initial_biscuits = 32) →
  (father_gift = 13) →
  (mother_gift = 15) →
  (brother_ate = 20) →
  initial_biscuits + father_gift + mother_gift - brother_ate = 40 := by
  sorry

end randy_biscuits_l190_190439


namespace constant_term_of_product_l190_190944

-- Define the polynomials
def poly1 (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + 7
def poly2 (x : ℝ) : ℝ := 4 * x^4 + 2 * x^2 + 10

-- Main statement: Prove that the constant term in the expansion of poly1 * poly2 is 70
theorem constant_term_of_product : (poly1 0) * (poly2 0) = 70 :=
by
  -- The proof would go here
  sorry

end constant_term_of_product_l190_190944


namespace monkey_count_l190_190019

theorem monkey_count (piles_1 piles_2 hands_1 hands_2 bananas_1_per_hand bananas_2_per_hand total_bananas_per_monkey : ℕ) 
  (h1 : piles_1 = 6) 
  (h2 : piles_2 = 4) 
  (h3 : hands_1 = 9) 
  (h4 : hands_2 = 12) 
  (h5 : bananas_1_per_hand = 14) 
  (h6 : bananas_2_per_hand = 9) 
  (h7 : total_bananas_per_monkey = 99) : 
  (piles_1 * hands_1 * bananas_1_per_hand + piles_2 * hands_2 * bananas_2_per_hand) / total_bananas_per_monkey = 12 := 
by 
  sorry

end monkey_count_l190_190019


namespace find_vector_c_l190_190080

def angle_equal_coordinates (c : ℝ × ℝ) : Prop :=
  let a : ℝ × ℝ := (1, 0)
  let b : ℝ × ℝ := (1, -Real.sqrt 3)
  let cos_angle_ab (u v : ℝ × ℝ) : ℝ :=
    (u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2))
  cos_angle_ab c a = cos_angle_ab c b

theorem find_vector_c :
  angle_equal_coordinates (Real.sqrt 3, -1) :=
sorry

end find_vector_c_l190_190080


namespace solve_equation_l190_190722

theorem solve_equation (x : ℝ) : (x + 3)^4 + (x + 1)^4 = 82 → x = 0 ∨ x = -4 :=
by
  sorry

end solve_equation_l190_190722


namespace floor_tiling_l190_190903

-- Define that n can be expressed as 7k for some integer k.
theorem floor_tiling (n : ℕ) (h : ∃ x : ℕ, n^2 = 7 * x) : ∃ k : ℕ, n = 7 * k := by
  sorry

end floor_tiling_l190_190903


namespace number_of_observations_l190_190382

theorem number_of_observations (n : ℕ) (h1 : 200 - 6 = 194) (h2 : 200 * n - n * 6 = n * 194) :
  n > 0 :=
by
  sorry

end number_of_observations_l190_190382


namespace laundry_loads_l190_190071

theorem laundry_loads (usual_price : ℝ) (sale_price : ℝ) (cost_per_load : ℝ) (total_loads_2_bottles : ℝ) :
  usual_price = 25 ∧ sale_price = 20 ∧ cost_per_load = 0.25 ∧ total_loads_2_bottles = (2 * sale_price) / cost_per_load →
  (total_loads_2_bottles / 2) = 80 :=
by
  sorry

end laundry_loads_l190_190071


namespace average_price_per_person_excluding_gratuity_l190_190042

def total_cost_with_gratuity : ℝ := 207.00
def gratuity_rate : ℝ := 0.15
def number_of_people : ℕ := 15

theorem average_price_per_person_excluding_gratuity :
  (total_cost_with_gratuity / (1 + gratuity_rate) / number_of_people) = 12.00 :=
by
  sorry

end average_price_per_person_excluding_gratuity_l190_190042


namespace largest_product_is_168_l190_190203

open Set

noncomputable def largest_product_from_set (s : Set ℤ) (n : ℕ) (result : ℤ) : Prop :=
  ∃ (a b c : ℤ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∀ (x y z : ℤ), x ∈ s → y ∈ s → z ∈ s → x ≠ y → y ≠ z → x ≠ z →
  x * y * z ≤ a * b * c ∧ a * b * c = result

theorem largest_product_is_168 :
  largest_product_from_set {-4, -3, 1, 3, 7, 8} 3 168 :=
sorry

end largest_product_is_168_l190_190203


namespace sin_cos_solution_count_l190_190191

-- Statement of the problem
theorem sin_cos_solution_count : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin (3 * x) = Real.cos (x / 2)) ∧ s.card = 6 := by
  sorry

end sin_cos_solution_count_l190_190191


namespace nontrivial_solution_exists_l190_190894

theorem nontrivial_solution_exists 
  (a b : ℤ) 
  (h_square_a : ∀ k : ℤ, a ≠ k^2) 
  (h_square_b : ∀ k : ℤ, b ≠ k^2) 
  (h_nontrivial : ∃ (x y z w : ℤ), x^2 - a * y^2 - b * z^2 + a * b * w^2 = 0 ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) : 
  ∃ (x y z : ℤ), x^2 - a * y^2 - b * z^2 = 0 ∧ (x, y, z) ≠ (0, 0, 0) :=
by
  sorry

end nontrivial_solution_exists_l190_190894


namespace incorrect_equation_l190_190444

theorem incorrect_equation (x : ℕ) (h : x + 2 * (12 - x) = 20) : 2 * (12 - x) - 20 ≠ x :=
by 
  sorry

end incorrect_equation_l190_190444


namespace min_value_reciprocal_l190_190143

theorem min_value_reciprocal (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_eq : 2 * a + b = 4) : 
  (∀ (x : ℝ), (∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = 4 -> x ≥ 1 / (2 * a * b)) -> x ≥ 1 / 2) := 
by
  sorry

end min_value_reciprocal_l190_190143


namespace calculate_altitude_l190_190061

-- Define the conditions
def Speed_up : ℕ := 18
def Speed_down : ℕ := 24
def Avg_speed : ℝ := 20.571428571428573

-- Define what we want to prove
theorem calculate_altitude : 
  2 * Speed_up * Speed_down / (Speed_up + Speed_down) = Avg_speed →
  (864 : ℝ) / 2 = 432 :=
by
  sorry

end calculate_altitude_l190_190061


namespace elvis_recording_time_l190_190936

theorem elvis_recording_time :
  ∀ (total_studio_time writing_time_per_song editing_time number_of_songs : ℕ),
  total_studio_time = 300 →
  writing_time_per_song = 15 →
  editing_time = 30 →
  number_of_songs = 10 →
  (total_studio_time - (number_of_songs * writing_time_per_song + editing_time)) / number_of_songs = 12 :=
by
  intros total_studio_time writing_time_per_song editing_time number_of_songs
  intros h1 h2 h3 h4
  sorry

end elvis_recording_time_l190_190936


namespace problem_statement_l190_190108

theorem problem_statement (x : ℝ) (h : x = Real.sqrt 3 + 1) : x^2 - 2*x + 1 = 3 :=
sorry

end problem_statement_l190_190108


namespace triangular_region_area_l190_190898

def line (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def y (x : ℝ) := x

theorem triangular_region_area : 
  ∀ (x y: ℝ),
  (y = line 1 2 x ∧ y = 3) ∨ 
  (y = line (-1) 8 x ∧ y = 3) ∨ 
  (y = line 1 2 x ∧ y = line (-1) 8 x)
  →
  ∃ (area: ℝ), area = 4.00 := 
by
  sorry

end triangular_region_area_l190_190898


namespace find_missing_number_l190_190907

theorem find_missing_number (n : ℝ) : n * 120 = 173 * 240 → n = 345.6 :=
by
  intros h
  sorry

end find_missing_number_l190_190907


namespace total_time_to_pump_540_gallons_l190_190015

-- Definitions for the conditions
def initial_rate : ℝ := 360  -- gallons per hour
def increased_rate : ℝ := 480 -- gallons per hour
def target_volume : ℝ := 540  -- total gallons
def first_interval : ℝ := 0.5 -- first 30 minutes as fraction of hour

-- Proof problem statement
theorem total_time_to_pump_540_gallons : 
  (first_interval * initial_rate) + ((target_volume - (first_interval * initial_rate)) / increased_rate) * 60 = 75 := by
  sorry

end total_time_to_pump_540_gallons_l190_190015


namespace solve_system_l190_190320

-- Define the conditions from the problem
def system_of_equations (x y : ℝ) : Prop :=
  (x = 4 * y) ∧ (x + 2 * y = -12)

-- Define the solution we want to prove
def solution (x y : ℝ) : Prop :=
  (x = -8) ∧ (y = -2)

-- State the theorem
theorem solve_system :
  ∃ x y : ℝ, system_of_equations x y ∧ solution x y :=
by 
  sorry

end solve_system_l190_190320


namespace problem_proof_l190_190609

theorem problem_proof (N : ℤ) (h : N / 5 = 4) : ((N - 10) * 3) - 18 = 12 :=
by
  -- proof goes here
  sorry

end problem_proof_l190_190609


namespace greatest_xy_value_l190_190640

theorem greatest_xy_value (x y : ℕ) (h1 : 7 * x + 4 * y = 140) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 112 :=
by
  sorry

end greatest_xy_value_l190_190640


namespace students_like_apple_chocolate_not_blueberry_l190_190959

theorem students_like_apple_chocolate_not_blueberry
  (n d a b c abc : ℕ)
  (h1 : n = 50)
  (h2 : d = 15)
  (h3 : a = 25)
  (h4 : b = 20)
  (h5 : c = 10)
  (h6 : abc = 5)
  (h7 : (n - d) = 35)
  (h8 : (55 - (a + b + c - abc)) = 35) :
  (20 - abc) = (15 : ℕ) :=
by
  sorry

end students_like_apple_chocolate_not_blueberry_l190_190959


namespace rectangle_symmetry_l190_190705

-- Definitions of symmetry properties
def isAxisymmetric (shape : Type) : Prop := sorry
def isCentrallySymmetric (shape : Type) : Prop := sorry

-- Specific shapes
def EquilateralTriangle : Type := sorry
def Parallelogram : Type := sorry
def Rectangle : Type := sorry
def RegularPentagon : Type := sorry

-- The theorem we want to prove
theorem rectangle_symmetry : 
  isAxisymmetric Rectangle ∧ isCentrallySymmetric Rectangle := sorry

end rectangle_symmetry_l190_190705


namespace jelly_cost_l190_190253

theorem jelly_cost (N B J : ℕ) (hN_gt_1 : N > 1) (h_cost_eq : N * (3 * B + 7 * J) = 252) : 7 * N * J = 168 := by
  sorry

end jelly_cost_l190_190253


namespace parallel_vectors_angle_l190_190330

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (1 / 2, Real.sin α)
noncomputable def vec_b (α : ℝ) : ℝ × ℝ := (Real.sin α, 1)

theorem parallel_vectors_angle (α : ℝ) (h_parallel : ∃ k : ℝ, k ≠ 0 ∧ (vec_a α).1 = k * (vec_b α).1 ∧ (vec_a α).2 = k * (vec_b α).2) (h_acute : 0 < α ∧ α < π / 2) :
  α = π / 4 :=
sorry

end parallel_vectors_angle_l190_190330


namespace find_x_collinear_l190_190797

theorem find_x_collinear (x : ℝ) (a b : ℝ × ℝ) (h_a : a = (2, 1)) (h_b : b = (x, -1)) 
  (h_collinear : ∃ k : ℝ, (a.1 - b.1, a.2 - b.2) = (k * b.1, k * b.2)) : x = -2 :=
by 
  -- the proof would go here
  sorry

end find_x_collinear_l190_190797


namespace rectangle_diagonals_equal_l190_190997

-- Define the properties of a rectangle
def is_rectangle (AB CD AD BC : ℝ) (diagonal1 diagonal2 : ℝ) : Prop :=
  AB = CD ∧ AD = BC ∧ diagonal1 = diagonal2

-- State the theorem to prove that the diagonals of a rectangle are equal
theorem rectangle_diagonals_equal (AB CD AD BC diagonal1 diagonal2 : ℝ) (h : is_rectangle AB CD AD BC diagonal1 diagonal2) :
  diagonal1 = diagonal2 :=
by
  sorry

end rectangle_diagonals_equal_l190_190997


namespace isosceles_triangles_possible_l190_190515

theorem isosceles_triangles_possible :
  ∃ (sticks : List ℕ), (sticks = [1, 1, 2, 2, 3, 3] ∧ 
    ∀ (a b c : ℕ), a ∈ sticks → b ∈ sticks → c ∈ sticks → 
    ((a + b > c ∧ b + c > a ∧ c + a > b) → a = b ∨ b = c ∨ c = a)) :=
sorry

end isosceles_triangles_possible_l190_190515


namespace sum_of_coefficients_l190_190283

-- Given polynomial definition
def P (x : ℝ) : ℝ := (1 + x - 3 * x^2) ^ 1965

-- Lean 4 statement for the proof problem
theorem sum_of_coefficients :
  P 1 = -1 :=
by
  -- Proof placeholder
  sorry

end sum_of_coefficients_l190_190283


namespace number_of_people_l190_190187

-- Definitions based on the conditions
def average_age (T : ℕ) (n : ℕ) := T / n = 30
def youngest_age := 3
def average_age_when_youngest_born (T : ℕ) (n : ℕ) := (T - youngest_age) / (n - 1) = 27

theorem number_of_people (T n : ℕ) (h1 : average_age T n) (h2 : average_age_when_youngest_born T n) : n = 7 :=
by
  sorry

end number_of_people_l190_190187


namespace reciprocal_difference_decreases_l190_190619

theorem reciprocal_difference_decreases (n : ℕ) (hn : n > 0) : 
  (1 / (n : ℝ) - 1 / (n + 1 : ℝ)) < (1 / (n * n : ℝ)) :=
by 
  sorry

end reciprocal_difference_decreases_l190_190619


namespace percentage_girls_l190_190748

theorem percentage_girls (x y : ℕ) (S₁ S₂ : ℕ)
  (h1 : S₁ = 22 * x)
  (h2 : S₂ = 47 * y)
  (h3 : (S₁ + S₂) / (x + y) = 41) :
  (x : ℝ) / (x + y) = 0.24 :=
sorry

end percentage_girls_l190_190748


namespace tan_proof_l190_190473

noncomputable def prove_tan_relation (α β : ℝ) : Prop :=
  2 * (Real.tan α) = 3 * (Real.tan β)

theorem tan_proof (α β : ℝ) (h : Real.tan (α - β) = (Real.sin (2*β)) / (5 - Real.cos (2*β))) : 
  prove_tan_relation α β :=
sorry

end tan_proof_l190_190473


namespace Karl_max_score_l190_190824

def max_possible_score : ℕ :=
  69

theorem Karl_max_score (minutes problems : ℕ) (n_points : ℕ → ℕ) (time_1_5 : ℕ) (time_6_10 : ℕ) (time_11_15 : ℕ)
    (h1 : minutes = 15) (h2 : problems = 15)
    (h3 : ∀ n, n = n_points n)
    (h4 : ∀ i, 1 ≤ i ∧ i ≤ 5 → time_1_5 = 1)
    (h5 : ∀ i, 6 ≤ i ∧ i ≤ 10 → time_6_10 = 2)
    (h6 : ∀ i, 11 ≤ i ∧ i ≤ 15 → time_11_15 = 3) : 
    max_possible_score = 69 :=
  by
  sorry

end Karl_max_score_l190_190824


namespace rebus_decrypt_correct_l190_190295

-- Definitions
def is_digit (d : ℕ) : Prop := 0 ≤ d ∧ d ≤ 9
def is_odd (d : ℕ) : Prop := is_digit d ∧ d % 2 = 1
def is_even (d : ℕ) : Prop := is_digit d ∧ d % 2 = 0

-- Variables representing ċharacters H, Ч (C), A, D, Y, E, F, B, K
variables (H C A D Y E F B K : ℕ)

-- Conditions
axiom H_odd : is_odd H
axiom C_even : is_even C
axiom A_even : is_even A
axiom D_odd : is_odd D
axiom Y_even : is_even Y
axiom E_even : is_even E
axiom F_odd : is_odd F
axiom B_digit : is_digit B
axiom K_odd : is_odd K

-- Correct answers
def H_val : ℕ := 5
def C_val : ℕ := 3
def A_val : ℕ := 2
def D_val : ℕ := 9
def Y_val : ℕ := 8
def E_val : ℕ := 8
def F_val : ℕ := 5
def B_any : ℕ := B
def K_val : ℕ := 5

-- Proof statement
theorem rebus_decrypt_correct : 
  H = H_val ∧
  C = C_val ∧
  A = A_val ∧
  D = D_val ∧
  Y = Y_val ∧
  E = E_val ∧
  F = F_val ∧
  K = K_val :=
sorry

end rebus_decrypt_correct_l190_190295


namespace y_value_l190_190745

theorem y_value {y : ℝ} (h1 : (0, 2) = (0, 2))
                (h2 : (3, y) = (3, y))
                (h3 : dist (0, 2) (3, y) = 10)
                (h4 : y > 0) :
                y = 2 + Real.sqrt 91 := by
  sorry

end y_value_l190_190745


namespace solve_ab_cd_l190_190788

theorem solve_ab_cd (a b c d : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a + b + d = -2) 
  (h3 : a + c + d = 5) 
  (h4 : b + c + d = 4) 
  : a * b + c * d = 26 / 9 := 
by {
  sorry
}

end solve_ab_cd_l190_190788


namespace unique_solution_real_l190_190440

theorem unique_solution_real {x y : ℝ} (h1 : x * (x + y)^2 = 9) (h2 : x * (y^3 - x^3) = 7) :
  x = 1 ∧ y = 2 :=
sorry

end unique_solution_real_l190_190440


namespace convert_spherical_coordinates_l190_190961

theorem convert_spherical_coordinates (
  ρ θ φ : ℝ
) (h1 : ρ = 5) (h2 : θ = 3 * Real.pi / 4) (h3 : φ = 9 * Real.pi / 4) : 
ρ = 5 ∧ 0 ≤ 7 * Real.pi / 4 ∧ 7 * Real.pi / 4 < 2 * Real.pi ∧ 0 ≤ Real.pi / 4 ∧ Real.pi / 4 ≤ Real.pi :=
by
  sorry

end convert_spherical_coordinates_l190_190961


namespace ratio_proof_l190_190111

variable (x y z : ℝ)
variable (h1 : y / z = 1 / 2)
variable (h2 : z / x = 2 / 3)
variable (h3 : x / y = 3 / 1)

theorem ratio_proof : (x / (y * z)) / (y / (z * x)) = 4 / 1 := 
  sorry

end ratio_proof_l190_190111


namespace smallest_number_divisible_l190_190248

theorem smallest_number_divisible
  (x : ℕ)
  (h : (x - 2) % 12 = 0 ∧ (x - 2) % 16 = 0 ∧ (x - 2) % 18 = 0 ∧ (x - 2) % 21 = 0 ∧ (x - 2) % 28 = 0) :
  x = 1010 :=
by
  sorry

end smallest_number_divisible_l190_190248


namespace min_value_condition_l190_190938

open Real

theorem min_value_condition 
  (m n : ℝ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : 2 * m + n = 1) : 
  (1 / m + 2 / n) ≥ 8 :=
sorry

end min_value_condition_l190_190938


namespace midpoint_trajectory_l190_190246

theorem midpoint_trajectory (x y : ℝ) :
  (∃ B C : ℝ × ℝ, B ≠ C ∧ (B.1^2 + B.2^2 = 25) ∧ (C.1^2 + C.2^2 = 25) ∧ 
                   (x, y) = ((B.1 + C.1)/2, (B.2 + C.2)/2) ∧ 
                   (B.1 - C.1)^2 + (B.2 - C.2)^2 = 36) →
  x^2 + y^2 = 16 :=
sorry

end midpoint_trajectory_l190_190246


namespace positive_integers_p_divisibility_l190_190725

theorem positive_integers_p_divisibility (p : ℕ) (hp : 0 < p) :
  (∃ n : ℕ, 0 < n ∧ p^n + 3^n ∣ p^(n+1) + 3^(n+1)) ↔ p = 3 ∨ p = 6 ∨ p = 15 :=
by sorry

end positive_integers_p_divisibility_l190_190725


namespace negation_of_no_vegetarian_students_eat_at_cafeteria_l190_190158

variable (Student : Type) 
variable (isVegetarian : Student → Prop)
variable (eatsAtCafeteria : Student → Prop)

theorem negation_of_no_vegetarian_students_eat_at_cafeteria :
  (∀ x, isVegetarian x → ¬ eatsAtCafeteria x) →
  (∃ x, isVegetarian x ∧ eatsAtCafeteria x) :=
by
  sorry

end negation_of_no_vegetarian_students_eat_at_cafeteria_l190_190158


namespace sum_of_series_l190_190347

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l190_190347


namespace percentage_k_equal_125_percent_j_l190_190475

theorem percentage_k_equal_125_percent_j
  (j k l m : ℝ)
  (h1 : 1.25 * j = (x / 100) * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 25 := 
sorry

end percentage_k_equal_125_percent_j_l190_190475


namespace chess_tournament_l190_190274

-- Define the number of chess amateurs
def num_amateurs : ℕ := 5

-- Define the number of games each amateur plays
def games_per_amateur : ℕ := 4

-- Define the total number of chess games possible
def total_games : ℕ := num_amateurs * (num_amateurs - 1) / 2

-- The main statement to prove
theorem chess_tournament : total_games = 10 := 
by
  -- here should be the proof, but according to the task, we use sorry to skip
  sorry

end chess_tournament_l190_190274


namespace range_of_t_for_point_in_upper_left_side_l190_190756

def point_in_upper_left_side_condition (x y : ℝ) : Prop :=
  x - y + 4 < 0

theorem range_of_t_for_point_in_upper_left_side :
  ∀ t : ℝ, point_in_upper_left_side_condition (-2) t ↔ t > 2 :=
by
  intros t
  unfold point_in_upper_left_side_condition
  simp
  sorry

end range_of_t_for_point_in_upper_left_side_l190_190756


namespace part1_condition1_implies_a_le_1_condition2_implies_a_le_2_condition3_implies_a_le_1_l190_190973

section Problem

-- Universal set is ℝ
def universal_set : Set ℝ := Set.univ

-- Set A
def set_A : Set ℝ := { x | x^2 - x - 6 ≤ 0 }

-- Set A complement in ℝ
def complement_A : Set ℝ := universal_set \ set_A

-- Set B
def set_B : Set ℝ := { x | (x - 4)/(x + 1) < 0 }

-- Set C
def set_C (a : ℝ) : Set ℝ := { x | 2 - a < x ∧ x < 2 + a }

-- Prove (complement_A ∩ set_B = (3, 4))
theorem part1 : (complement_A ∩ set_B) = { x | 3 < x ∧ x < 4 } :=
  sorry

-- Assume a definition for real number a (non-negative)
variable (a : ℝ)

-- Prove range of a given the conditions
-- Condition 1: A ∩ C = C implies a ≤ 1
theorem condition1_implies_a_le_1 (h : set_A ∩ set_C a = set_C a) : a ≤ 1 :=
  sorry

-- Condition 2: B ∪ C = B implies a ≤ 2
theorem condition2_implies_a_le_2 (h : set_B ∪ set_C a = set_B) : a ≤ 2 :=
  sorry

-- Condition 3: C ⊆ (A ∩ B) implies a ≤ 1
theorem condition3_implies_a_le_1 (h : set_C a ⊆ set_A ∩ set_B) : a ≤ 1 :=
  sorry

end Problem

end part1_condition1_implies_a_le_1_condition2_implies_a_le_2_condition3_implies_a_le_1_l190_190973


namespace max_value_of_4x_plus_3y_l190_190345

theorem max_value_of_4x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 18 * x + 8 * y + 10) :
  4 * x + 3 * y ≤ 45 :=
sorry

end max_value_of_4x_plus_3y_l190_190345


namespace min_omega_value_l190_190530

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega_value (ω : ℝ) (φ : ℝ) (h_ω_pos : ω > 0)
  (h_even : ∀ x : ℝ, f ω φ x = f ω φ (-x))
  (h_symmetry : f ω φ 1 = 0 ∧ ∀ x : ℝ, f ω φ (1 + x) = - f ω φ (1 - x)) :
  ω = Real.pi / 2 :=
by
  sorry

end min_omega_value_l190_190530


namespace biased_die_probability_l190_190118

theorem biased_die_probability (P2 : ℝ) (h1 : P2 ≠ 1 / 6) (h2 : 3 * P2 * (1 - P2) ^ 2 = 1 / 4) : 
  P2 = 0.211 :=
sorry

end biased_die_probability_l190_190118


namespace connected_distinct_points_with_slope_change_l190_190335

-- Defining the cost function based on the given conditions
def cost_function (n : ℕ) : ℕ := 
  if n <= 10 then 20 * n else 18 * n

-- The main theorem to prove the nature of the graph as described in the problem
theorem connected_distinct_points_with_slope_change : 
  (∀ n, (1 ≤ n ∧ n ≤ 20) → 
    (∃ k, cost_function n = k ∧ 
    (n <= 10 → cost_function n = 20 * n) ∧ 
    (n > 10 → cost_function n = 18 * n))) ∧
  (∃ n, n = 10 ∧ cost_function n = 200 ∧ cost_function (n + 1) = 198) :=
sorry

end connected_distinct_points_with_slope_change_l190_190335


namespace quadratic_roots_l190_190299

theorem quadratic_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : a - b + c = 0) (h3 : (b^2 - 4 * a * c) = 0) : 2 * a - b = 0 :=
by {
  sorry
}

end quadratic_roots_l190_190299


namespace total_trophies_correct_l190_190167

-- Define the current number of Michael's trophies
def michael_current_trophies : ℕ := 30

-- Define the number of trophies Michael will have in three years
def michael_trophies_in_three_years : ℕ := michael_current_trophies + 100

-- Define the number of trophies Jack will have in three years
def jack_trophies_in_three_years : ℕ := 10 * michael_current_trophies

-- Define the total number of trophies Jack and Michael will have after three years
def total_trophies_in_three_years : ℕ := michael_trophies_in_three_years + jack_trophies_in_three_years

-- Prove that the total number of trophies after three years is 430
theorem total_trophies_correct : total_trophies_in_three_years = 430 :=
by
  sorry -- proof is omitted

end total_trophies_correct_l190_190167


namespace octal_to_base5_conversion_l190_190461

-- Define the octal to decimal conversion
def octalToDecimal (n : ℕ) : ℕ :=
  2 * 8^3 + 0 * 8^2 + 1 * 8^1 + 1 * 8^0

-- Define the base-5 number
def base5Representation : ℕ := 13113

-- Theorem statement
theorem octal_to_base5_conversion :
  octalToDecimal 2011 = base5Representation := 
sorry

end octal_to_base5_conversion_l190_190461


namespace value_of_s_in_base_b_l190_190796

noncomputable def b : ℕ :=
  10

def fourteen_in_b (b : ℕ) : ℕ :=
  b + 4

def seventeen_in_b (b : ℕ) : ℕ :=
  b + 7

def eighteen_in_b (b : ℕ) : ℕ :=
  b + 8

def five_thousand_four_and_four_in_b (b : ℕ) : ℕ :=
  5 * b ^ 3 + 4 * b ^ 2 + 4

def product_in_base_b_equals (b : ℕ) : Prop :=
  (fourteen_in_b b) * (seventeen_in_b b) * (eighteen_in_b b) = five_thousand_four_and_four_in_b b

def s_in_base_b (b : ℕ) : ℕ :=
  fourteen_in_b b + seventeen_in_b b + eighteen_in_b b

theorem value_of_s_in_base_b (b : ℕ) (h : product_in_base_b_equals b) : s_in_base_b b = 49 := by
  sorry

end value_of_s_in_base_b_l190_190796


namespace total_payment_correct_l190_190399

def payment_y : ℝ := 318.1818181818182
def payment_ratio : ℝ := 1.2
def payment_x : ℝ := payment_ratio * payment_y
def total_payment : ℝ := payment_x + payment_y

theorem total_payment_correct :
  total_payment = 700.00 :=
sorry

end total_payment_correct_l190_190399


namespace minimize_quadratic_l190_190692

theorem minimize_quadratic : ∃ x : ℝ, x = 6 ∧ ∀ y : ℝ, (y - 6)^2 ≥ (6 - 6)^2 := by
  sorry

end minimize_quadratic_l190_190692


namespace wheel_radius_l190_190843

theorem wheel_radius 
(D: ℝ) (N: ℕ) (r: ℝ) 
(hD: D = 88 * 1000) 
(hN: N = 1000) 
(hC: 2 * Real.pi * r * N = D) : 
r = 88 / (2 * Real.pi) :=
by
  sorry

end wheel_radius_l190_190843


namespace two_same_color_probability_l190_190775

-- Definitions based on the given conditions
def total_balls := 5
def black_balls := 3
def red_balls := 2

-- Definition for drawing two balls at random
def draw_two_same_color_probability : ℚ :=
  let total_ways := Nat.choose total_balls 2
  let black_pairs := Nat.choose black_balls 2
  let red_pairs := Nat.choose red_balls 2
  (black_pairs + red_pairs) / total_ways

-- Statement of the theorem
theorem two_same_color_probability :
  draw_two_same_color_probability = 2 / 5 :=
  sorry

end two_same_color_probability_l190_190775


namespace shaded_area_of_pattern_l190_190759

theorem shaded_area_of_pattern (d : ℝ) (L : ℝ) (n : ℕ) (r : ℝ) (A : ℝ) : 
  d = 3 → 
  L = 24 → 
  n = 16 → 
  r = 3 / 2 → 
  (A = 18 * Real.pi) :=
by
  intro hd
  intro hL
  intro hn
  intro hr
  sorry

end shaded_area_of_pattern_l190_190759


namespace student_difference_l190_190627

theorem student_difference 
  (C1 : ℕ) (x : ℕ)
  (hC1 : C1 = 25)
  (h_total : C1 + (C1 - x) + (C1 - 2 * x) + (C1 - 3 * x) + (C1 - 4 * x) = 105) : 
  x = 2 := 
by
  sorry

end student_difference_l190_190627


namespace molecular_weight_ammonia_l190_190465

def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.008
def count_N : ℕ := 1
def count_H : ℕ := 3

theorem molecular_weight_ammonia :
  (count_N * atomic_weight_N) + (count_H * atomic_weight_H) = 17.034 :=
by
  sorry

end molecular_weight_ammonia_l190_190465


namespace g_h_2_eq_583_l190_190381

def g (x : ℝ) : ℝ := 3*x^2 - 5

def h (x : ℝ) : ℝ := -2*x^3 + 2

theorem g_h_2_eq_583 : g (h 2) = 583 :=
by
  sorry

end g_h_2_eq_583_l190_190381


namespace ac_length_l190_190222

theorem ac_length (a b c d e : ℝ)
  (h1 : b - a = 5)
  (h2 : c - b = 2 * (d - c))
  (h3 : e - d = 4)
  (h4 : e - a = 18) :
  d - a = 11 :=
by
  sorry

end ac_length_l190_190222


namespace part1_part2_l190_190900

-- The quadratic equation of interest
def quadratic_eq (k x : ℝ) : ℝ :=
  x^2 + (2 * k - 1) * x + k^2 - k

-- Part 1: Proof that the equation has two distinct real roots
theorem part1 (k : ℝ) : (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic_eq k x1 = 0 ∧ quadratic_eq k x2 = 0) := 
  sorry

-- Part 2: Given x = 2 is a root, prove the value of the expression
theorem part2 (k : ℝ) (h : quadratic_eq k 2 = 0) : -2 * k^2 - 6 * k - 5 = -1 :=
  sorry

end part1_part2_l190_190900


namespace mowing_field_time_l190_190268

theorem mowing_field_time (h1 : (1 / 28 : ℝ) = (3 / 84 : ℝ))
                         (h2 : (1 / 84 : ℝ) = (1 / 84 : ℝ))
                         (h3 : (1 / 28 + 1 / 84 : ℝ) = (1 / 21 : ℝ)) :
                         21 = 1 / ((1 / 28) + (1 / 84)) := 
by {
  sorry
}

end mowing_field_time_l190_190268


namespace mean_of_eight_numbers_l190_190533

theorem mean_of_eight_numbers (sum_of_numbers : ℚ) (h : sum_of_numbers = 3/4) : 
  sum_of_numbers / 8 = 3/32 := by
  sorry

end mean_of_eight_numbers_l190_190533


namespace bulb_probability_gt4000_l190_190055

-- Definitions given in conditions
def P_X : ℝ := 0.60
def P_Y : ℝ := 0.40
def P_gt4000_X : ℝ := 0.59
def P_gt4000_Y : ℝ := 0.65

-- The proof statement
theorem bulb_probability_gt4000 : 
  (P_X * P_gt4000_X + P_Y * P_gt4000_Y) = 0.614 :=
  by
  sorry

end bulb_probability_gt4000_l190_190055


namespace linear_function_above_x_axis_l190_190536

theorem linear_function_above_x_axis (a : ℝ) :
  (-1 < a ∧ a < 2 ∧ a ≠ 0) ↔
  (∀ x, -2 ≤ x ∧ x ≤ 1 → ax + a + 2 > 0) :=
sorry

end linear_function_above_x_axis_l190_190536


namespace smallest_multiple_of_9_and_6_is_18_l190_190035

theorem smallest_multiple_of_9_and_6_is_18 :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) :=
sorry

end smallest_multiple_of_9_and_6_is_18_l190_190035


namespace sum_of_decimals_as_fraction_l190_190054

theorem sum_of_decimals_as_fraction :
  (0.2 : ℝ) + (0.03 : ℝ) + (0.004 : ℝ) + (0.0006 : ℝ) + (0.00007 : ℝ) + (0.000008 : ℝ) + (0.0000009 : ℝ) = 
  (2340087 / 10000000 : ℝ) :=
sorry

end sum_of_decimals_as_fraction_l190_190054


namespace lamp_cost_l190_190915

def saved : ℕ := 500
def couch : ℕ := 750
def table : ℕ := 100
def remaining_owed : ℕ := 400

def total_cost_without_lamp : ℕ := couch + table

theorem lamp_cost :
  total_cost_without_lamp - saved + lamp = remaining_owed → lamp = 50 := by
  sorry

end lamp_cost_l190_190915


namespace quadratic_root_condition_l190_190446

theorem quadratic_root_condition (d : ℝ) :
  (∀ x, x^2 + 7 * x + d = 0 → x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → d = 9.8 :=
by
  intro h
  sorry

end quadratic_root_condition_l190_190446


namespace original_price_of_cycle_l190_190075

theorem original_price_of_cycle (SP : ℝ) (P : ℝ) (loss_percent : ℝ) 
  (h_loss : loss_percent = 18) 
  (h_SP : SP = 1148) 
  (h_eq : SP = (1 - loss_percent / 100) * P) : 
  P = 1400 := 
by 
  sorry

end original_price_of_cycle_l190_190075


namespace value_of_a_l190_190772

theorem value_of_a :
  ∀ (a : ℤ) (BO CO : ℤ), 
  BO = 2 → 
  CO = 2 * BO → 
  |a + 3| = CO → 
  a < 0 → 
  a = -7 := by
  intros a BO CO hBO hCO hAbs ha_neg
  sorry

end value_of_a_l190_190772


namespace height_difference_l190_190872

def burj_khalifa_height : ℝ := 830
def sears_tower_height : ℝ := 527

theorem height_difference : burj_khalifa_height - sears_tower_height = 303 := 
by
  sorry

end height_difference_l190_190872


namespace unique_integer_solution_l190_190249

theorem unique_integer_solution (x : ℤ) : x^3 + (x + 1)^3 + (x + 2)^3 = (x + 3)^3 ↔ x = 3 := by
  sorry

end unique_integer_solution_l190_190249


namespace new_paint_intensity_l190_190351

theorem new_paint_intensity : 
  let I_original : ℝ := 0.5
  let I_added : ℝ := 0.2
  let replacement_fraction : ℝ := 1 / 3
  let remaining_fraction : ℝ := 2 / 3
  let I_new := remaining_fraction * I_original + replacement_fraction * I_added
  I_new = 0.4 :=
by
  -- sorry is used to skip the actual proof
  sorry

end new_paint_intensity_l190_190351


namespace intervals_of_monotonicity_range_of_a_l190_190025

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * log x

theorem intervals_of_monotonicity (h : ∀ x, 0 < x → x ≠ e → f (-2) x = -2 * x + x * log x) :
  ((∀ x, 0 < x ∧ x < exp 1 → deriv (f (-2)) x < 0) ∧ (∀ x, x > exp 1 → deriv (f (-2)) x > 0)) :=
sorry

theorem range_of_a (h : ∀ x, e ≤ x → deriv (f a) x ≥ 0) : a ≥ -2 :=
sorry

end intervals_of_monotonicity_range_of_a_l190_190025


namespace invertible_my_matrix_l190_190496

def my_matrix : Matrix (Fin 2) (Fin 2) ℚ := ![![4, 5], ![-2, 9]]

noncomputable def inverse_of_my_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  Matrix.det my_matrix • Matrix.adjugate my_matrix

theorem invertible_my_matrix :
  inverse_of_my_matrix = (1 / 46 : ℚ) • ![![9, -5], ![2, 4]] :=
by
  sorry

end invertible_my_matrix_l190_190496


namespace quadratic_inequality_solution_l190_190563

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 50 * x + 601 ≤ 9} = {x : ℝ | 19.25545 ≤ x ∧ x ≤ 30.74455} :=
by 
  sorry

end quadratic_inequality_solution_l190_190563


namespace three_digit_numbers_divisible_by_17_l190_190057

theorem three_digit_numbers_divisible_by_17 : ∃ (n : ℕ), n = 53 ∧ ∀ k, 100 <= 17 * k ∧ 17 * k <= 999 ↔ (6 <= k ∧ k <= 58) :=
by
  sorry

end three_digit_numbers_divisible_by_17_l190_190057


namespace trigonometric_identity_l190_190926

noncomputable def tan_sum (alpha : ℝ) : Prop :=
  Real.tan (alpha + Real.pi / 4) = 2

noncomputable def trigonometric_expression (alpha : ℝ) : ℝ :=
  (Real.sin alpha + 2 * Real.cos alpha) / (Real.sin alpha - 2 * Real.cos alpha)

theorem trigonometric_identity (alpha : ℝ) (h : tan_sum alpha) : 
  trigonometric_expression alpha = -7 / 5 :=
sorry

end trigonometric_identity_l190_190926


namespace number_made_l190_190483

theorem number_made (x y : ℕ) (h1 : x + y = 24) (h2 : x = 11) : 7 * x + 5 * y = 142 := by
  sorry

end number_made_l190_190483


namespace reciprocal_eq_self_l190_190578

open Classical

theorem reciprocal_eq_self (a : ℝ) (h : a = 1 / a) : a = 1 ∨ a = -1 := 
sorry

end reciprocal_eq_self_l190_190578


namespace range_of_m_l190_190328

def f (x : ℝ) : ℝ := -x^3 - 2*x^2 + 4*x

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ m^2 - 14 * m) ↔ 3 ≤ m ∧ m ≤ 11 :=
by
  sorry

end range_of_m_l190_190328


namespace overtime_hourly_rate_l190_190139

theorem overtime_hourly_rate
  (hourly_rate_first_40_hours: ℝ)
  (hours_first_40: ℝ)
  (gross_pay: ℝ)
  (overtime_hours: ℝ)
  (total_pay_first_40: ℝ := hours_first_40 * hourly_rate_first_40_hours)
  (pay_overtime: ℝ := gross_pay - total_pay_first_40)
  (hourly_rate_overtime: ℝ := pay_overtime / overtime_hours)
  (h1: hourly_rate_first_40_hours = 11.25)
  (h2: hours_first_40 = 40)
  (h3: gross_pay = 622)
  (h4: overtime_hours = 10.75) :
  hourly_rate_overtime = 16 := 
by
  sorry

end overtime_hourly_rate_l190_190139


namespace minimum_value_frac_sum_l190_190674

-- Define the statement problem C and proof outline skipping the steps
theorem minimum_value_frac_sum (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 :=
by
  -- Proof is to be constructed here
  sorry

end minimum_value_frac_sum_l190_190674


namespace absolute_difference_l190_190429

theorem absolute_difference : |8 - 3^2| - |4^2 - 6*3| = -1 := by
  sorry

end absolute_difference_l190_190429


namespace solve_trig_equation_l190_190919

theorem solve_trig_equation (x : ℝ) : 
  2 * Real.cos (13 * x) + 3 * Real.cos (3 * x) + 3 * Real.cos (5 * x) - 8 * Real.cos x * (Real.cos (4 * x))^3 = 0 ↔ 
  ∃ (k : ℤ), x = (k * Real.pi) / 12 :=
sorry

end solve_trig_equation_l190_190919


namespace rectangle_length_width_ratio_l190_190832

-- Define the side lengths of the small squares and the large square
variables (s : ℝ)

-- Define the dimensions of the large square and the rectangle
def large_square_side : ℝ := 5 * s
def rectangle_length : ℝ := 5 * s
def rectangle_width : ℝ := s

-- State and prove the theorem
theorem rectangle_length_width_ratio : rectangle_length s / rectangle_width s = 5 :=
by sorry

end rectangle_length_width_ratio_l190_190832


namespace necessary_but_not_sufficient_condition_l190_190100

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient_condition (a : ℝ) : (a ∈ M → a ∈ N) ∧ ¬(a ∈ N → a ∈ M) := 
  by 
    sorry

end necessary_but_not_sufficient_condition_l190_190100


namespace customers_added_during_lunch_rush_l190_190953

noncomputable def initial_customers := 29.0
noncomputable def total_customers_after_lunch_rush := 83.0
noncomputable def expected_customers_added := 54.0

theorem customers_added_during_lunch_rush :
  (total_customers_after_lunch_rush - initial_customers) = expected_customers_added :=
by
  sorry

end customers_added_during_lunch_rush_l190_190953


namespace find_numbers_l190_190355

theorem find_numbers 
  (x y z : ℕ) 
  (h1 : y = 2 * x - 3) 
  (h2 : x + y = 51) 
  (h3 : z = 4 * x - y) : 
  x = 18 ∧ y = 33 ∧ z = 39 :=
by sorry

end find_numbers_l190_190355


namespace number_of_real_solutions_l190_190546

theorem number_of_real_solutions (floor : ℝ → ℤ) 
  (h_floor : ∀ x, floor x = ⌊x⌋)
  (h_eq : ∀ x, 9 * x^2 - 45 * floor (x^2 - 1) + 94 = 0) :
  ∃ n : ℕ, n = 2 :=
by
  sorry

end number_of_real_solutions_l190_190546


namespace intersection_point_not_on_x_3_l190_190638

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8*x + 15) / (3*x - 6)
noncomputable def g (x : ℝ) : ℝ := (-1/3 * x^2 + 6*x - 6) / (x - 2)

theorem intersection_point_not_on_x_3 : 
  ∃ x y : ℝ, (x ≠ 3) ∧ (f x = g x) ∧ (y = f x) ∧ (x = 11/3 ∧ y = -11/3) :=
by
  sorry

end intersection_point_not_on_x_3_l190_190638


namespace find_a_b_extreme_values_l190_190960

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + b * x - (2/3)

theorem find_a_b_extreme_values : 
  ∃ (a b : ℝ), 
    (a = -2) ∧ 
    (b = 3) ∧ 
    (f 1 (-2) 3 = 2/3) ∧ 
    (f 3 (-2) 3 = -2/3) :=
by
  sorry

end find_a_b_extreme_values_l190_190960


namespace total_paintable_area_is_2006_l190_190989

-- Define the dimensions of the bedrooms and the hallway
def bedroom_length := 14
def bedroom_width := 11
def bedroom_height := 9

def hallway_length := 20
def hallway_width := 7
def hallway_height := 9

def num_bedrooms := 4
def doorway_window_area := 70

-- Compute the areas of the bedroom walls and the hallway walls
def bedroom_wall_area : ℕ :=
  2 * (bedroom_length * bedroom_height) +
  2 * (bedroom_width * bedroom_height)

def paintable_bedroom_wall_area : ℕ :=
  bedroom_wall_area - doorway_window_area

def total_paintable_bedroom_area : ℕ :=
  num_bedrooms * paintable_bedroom_wall_area

def hallway_wall_area : ℕ :=
  2 * (hallway_length * hallway_height) +
  2 * (hallway_width * hallway_height)

-- Compute the total paintable area
def total_paintable_area : ℕ :=
  total_paintable_bedroom_area + hallway_wall_area

-- Theorem stating the total paintable area is 2006 sq ft
theorem total_paintable_area_is_2006 : total_paintable_area = 2006 := 
  by
    unfold total_paintable_area
    rw [total_paintable_bedroom_area, paintable_bedroom_wall_area, bedroom_wall_area]
    rw [hallway_wall_area]
    norm_num
    sorry -- Proof omitted

end total_paintable_area_is_2006_l190_190989


namespace symmetry_x_y_axis_symmetry_line_y_neg1_l190_190811

-- Define point P
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P : Point := { x := 1, y := 2 }

-- Condition for symmetry with respect to x-axis
def symmetric_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Condition for symmetry with respect to the line y = -1
def symmetric_line_y_neg1 (p : Point) : Point :=
  { x := p.x, y := 2 * 1 - p.y - 1 }

-- Theorem statements
theorem symmetry_x_y_axis : symmetric_x P = { x := 1, y := -2 } := sorry
theorem symmetry_line_y_neg1 : symmetric_line_y_neg1 { x := 1, y := -2 } = { x := 1, y := 3 } := sorry

end symmetry_x_y_axis_symmetry_line_y_neg1_l190_190811


namespace carlos_class_number_l190_190136

theorem carlos_class_number (b : ℕ) :
  (100 < b ∧ b < 200) ∧
  (b + 2) % 4 = 0 ∧
  (b + 3) % 5 = 0 ∧
  (b + 4) % 6 = 0 →
  b = 122 ∨ b = 182 :=
by
  -- The proof implementation goes here
  sorry

end carlos_class_number_l190_190136


namespace increasing_iff_positive_difference_l190_190163

variable (a : ℕ → ℝ) (d : ℝ)

def arithmetic_sequence (aₙ : ℕ → ℝ) (d : ℝ) := ∃ (a₁ : ℝ), ∀ n : ℕ, aₙ n = a₁ + n * d

theorem increasing_iff_positive_difference (a : ℕ → ℝ) (d : ℝ) (h : arithmetic_sequence a d) :
  (∀ n, a (n+1) > a n) ↔ d > 0 :=
by
  sorry

end increasing_iff_positive_difference_l190_190163


namespace quadratic_equation_m_value_l190_190499

theorem quadratic_equation_m_value (m : ℝ) (h : m ≠ 2) : m = -2 :=
by
  -- details of the proof go here
  sorry

end quadratic_equation_m_value_l190_190499


namespace num_women_in_luxury_suite_l190_190549

theorem num_women_in_luxury_suite (total_passengers : ℕ) (pct_women : ℕ) (pct_women_luxury : ℕ)
  (h_total_passengers : total_passengers = 300)
  (h_pct_women : pct_women = 50)
  (h_pct_women_luxury : pct_women_luxury = 15) :
  (total_passengers * pct_women / 100) * pct_women_luxury / 100 = 23 := 
by
  sorry

end num_women_in_luxury_suite_l190_190549


namespace hours_worked_each_day_l190_190977

-- Definitions based on problem conditions
def total_hours_worked : ℝ := 8.0
def number_of_days_worked : ℝ := 4.0

-- Theorem statement to prove the number of hours worked each day
theorem hours_worked_each_day :
  total_hours_worked / number_of_days_worked = 2.0 :=
sorry

end hours_worked_each_day_l190_190977


namespace simplify_fraction_l190_190937

theorem simplify_fraction : (90 : ℚ) / (126 : ℚ) = 5 / 7 := 
by
  sorry

end simplify_fraction_l190_190937


namespace no_positive_integer_has_product_as_perfect_square_l190_190795

theorem no_positive_integer_has_product_as_perfect_square:
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, n * (n + 1) = k * k :=
by
  sorry

end no_positive_integer_has_product_as_perfect_square_l190_190795


namespace one_meter_eq_jumps_l190_190010

theorem one_meter_eq_jumps 
  (x y a b p q s t : ℝ) 
  (h1 : x * hops = y * skips)
  (h2 : a * jumps = b * hops)
  (h3 : p * skips = q * leaps)
  (h4 : s * leaps = t * meters) :
  1 * meters = (sp * x * a / (tq * y * b)) * jumps :=
sorry

end one_meter_eq_jumps_l190_190010


namespace compute_a_plus_b_l190_190304

theorem compute_a_plus_b (a b : ℝ) (h : ∃ (u v w : ℕ), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ u + v + w = 8 ∧ u * v * w = b ∧ u * v + v * w + w * u = a) : 
  a + b = 27 :=
by
  -- The proof is omitted.
  sorry

end compute_a_plus_b_l190_190304


namespace change_received_l190_190752

theorem change_received (basic_cost : ℕ) (scientific_cost : ℕ) (graphing_cost : ℕ) (total_money : ℕ) :
  basic_cost = 8 →
  scientific_cost = 2 * basic_cost →
  graphing_cost = 3 * scientific_cost →
  total_money = 100 →
  (total_money - (basic_cost + scientific_cost + graphing_cost)) = 28 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end change_received_l190_190752


namespace quotient_remainder_increase_l190_190089

theorem quotient_remainder_increase (a b q r q' r' : ℕ) (hb : b ≠ 0) 
    (h1 : a = b * q + r) (h2 : 0 ≤ r) (h3 : r < b) (h4 : 3 * a = 3 * b * q' + r') 
    (h5 : 0 ≤ r') (h6 : r' < 3 * b) :
    q' = q ∧ r' = 3 * r := by
  sorry

end quotient_remainder_increase_l190_190089


namespace xiao_ming_arrival_time_l190_190703

def left_home (departure_time : String) : Prop :=
  departure_time = "6:55"

def time_spent (duration : Nat) : Prop :=
  duration = 30

def arrival_time (arrival : String) : Prop :=
  arrival = "7:25"

theorem xiao_ming_arrival_time :
  left_home "6:55" → time_spent 30 → arrival_time "7:25" :=
by sorry

end xiao_ming_arrival_time_l190_190703


namespace primes_sum_product_condition_l190_190376

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem primes_sum_product_condition (m n p : ℕ) (hm : is_prime m) (hn : is_prime n) (hp : is_prime p)  
  (h : m * n * p = 5 * (m + n + p)) : 
  m^2 + n^2 + p^2 = 78 :=
sorry

end primes_sum_product_condition_l190_190376


namespace least_even_perimeter_l190_190212

def triangle_perimeter (a b c : ℕ) : ℕ := a + b + c

theorem least_even_perimeter
  (a b : ℕ) (h1 : a = 24) (h2 : b = 37) (c : ℕ)
  (h3 : c > b) (h4 : a + b > c)
  (h5 : ∃ k : ℕ, k * 2 = triangle_perimeter a b c) :
  triangle_perimeter a b c = 100 :=
sorry

end least_even_perimeter_l190_190212


namespace solve_quadratic_eq_l190_190200

theorem solve_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := sorry

end solve_quadratic_eq_l190_190200


namespace share_of_B_in_profit_l190_190198

variable {D : ℝ} (hD_pos : 0 < D)

def investment (D : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let C := 2.5 * D
  let B := 1.25 * D
  let A := 5 * B
  (A, B, C, D)

def totalInvestment (A B C D : ℝ) : ℝ :=
  A + B + C + D

theorem share_of_B_in_profit (D : ℝ) (profit : ℝ) (hD : 0 < D)
  (h_profit : profit = 8000) :
  let ⟨A, B, C, D⟩ := investment D
  B / totalInvestment A B C D * profit = 1025.64 :=
by
  sorry

end share_of_B_in_profit_l190_190198


namespace findNumberOfIntegers_l190_190113

def arithmeticSeq (a d n : ℕ) : ℕ :=
  a + d * n

def isInSeq (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ 33 ∧ n = arithmeticSeq 1 3 k

def validInterval (n : ℕ) : Bool :=
  (n + 1) / 3 % 2 = 1

theorem findNumberOfIntegers :
  ∃ count : ℕ, count = 66 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 ∧ ¬isInSeq n → validInterval n = true) :=
sorry

end findNumberOfIntegers_l190_190113


namespace isosceles_triangle_perimeter_l190_190171

theorem isosceles_triangle_perimeter (a b : ℕ) (h₀ : a = 3 ∨ a = 4) (h₁ : b = 3 ∨ b = 4) (h₂ : a ≠ b) :
  (a = 3 ∧ b = 4 ∧ 4 ∈ [b]) ∨ (a = 4 ∧ b = 3 ∧ 4 ∈ [a]) → 
  (a + a + b = 10) ∨ (a + b + b = 11) :=
by
  sorry

end isosceles_triangle_perimeter_l190_190171


namespace max_non_real_roots_l190_190489

theorem max_non_real_roots (n : ℕ) (h_odd : n % 2 = 1) :
  (∃ (A B : ℕ → ℕ) (h_turns : ∀ i < 3 * n, A i + B i = 1),
    (∀ i, (A i + B (i + 1)) % 3 = 0) →
    ∃ k, ∀ m, ∃ j < n, j % 2 = 1 → j + m * 2 ≤ 2 * k + j - m)
  → (∃ k, k = (n + 1) / 2) :=
sorry

end max_non_real_roots_l190_190489


namespace no_12_term_geometric_seq_in_1_to_100_l190_190714

theorem no_12_term_geometric_seq_in_1_to_100 :
  ¬ ∃ (s : Fin 12 → Set ℕ),
    (∀ i, ∃ (a q : ℕ), (s i = {a * q^n | n : ℕ}) ∧ (∀ x ∈ s i, 1 ≤ x ∧ x ≤ 100)) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → ∃ i, n ∈ s i) := 
sorry

end no_12_term_geometric_seq_in_1_to_100_l190_190714


namespace age_ratio_l190_190425

-- Define the conditions
def ArunCurrentAgeAfter6Years (A: ℕ) : Prop := A + 6 = 36
def DeepakCurrentAge : ℕ := 42

-- Define the goal statement
theorem age_ratio (A: ℕ) (hc: ArunCurrentAgeAfter6Years A) : A / gcd A DeepakCurrentAge = 5 ∧ DeepakCurrentAge / gcd A DeepakCurrentAge = 7 :=
by
  sorry

end age_ratio_l190_190425


namespace julie_read_yesterday_l190_190706

variable (x : ℕ)
variable (y : ℕ := 2 * x)
variable (remaining_pages_after_two_days : ℕ := 120 - (x + y))

theorem julie_read_yesterday :
  (remaining_pages_after_two_days / 2 = 42) -> (x = 12) :=
by
  sorry

end julie_read_yesterday_l190_190706


namespace real_root_of_system_l190_190293

theorem real_root_of_system :
  (∃ x : ℝ, x^3 + 9 = 0 ∧ x + 3 = 0) ↔ x = -3 := 
by 
  sorry

end real_root_of_system_l190_190293


namespace power_comparison_l190_190477

noncomputable
def compare_powers : Prop := 
  1.5^(1 / 3.1) < 2^(1 / 3.1) ∧ 2^(1 / 3.1) < 2^(3.1)

theorem power_comparison : compare_powers :=
by
  sorry

end power_comparison_l190_190477


namespace g_7_eq_98_l190_190712

noncomputable def g : ℕ → ℝ := sorry

axiom g_0 : g 0 = 0
axiom g_1 : g 1 = 2
axiom functional_equation (m n : ℕ) (h : m ≥ n) : g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2

theorem g_7_eq_98 : g 7 = 98 :=
sorry

end g_7_eq_98_l190_190712


namespace quadratic_solution_l190_190831

theorem quadratic_solution (x : ℝ) :
  (x^2 + 2 * x = 0) ↔ (x = 0 ∨ x = -2) :=
by
  sorry

end quadratic_solution_l190_190831


namespace max_x_lcm_15_21_105_l190_190691

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l190_190691


namespace balls_in_boxes_l190_190415

def num_ways_to_partition_6_in_4_parts : ℕ :=
  -- The different partitions of 6: (6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0),
  -- (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)
  ([
    [6, 0, 0, 0],
    [5, 1, 0, 0],
    [4, 2, 0, 0],
    [4, 1, 1, 0],
    [3, 3, 0, 0],
    [3, 2, 1, 0],
    [3, 1, 1, 1],
    [2, 2, 2, 0],
    [2, 2, 1, 1]
  ]).length

theorem balls_in_boxes : num_ways_to_partition_6_in_4_parts = 9 := by
  sorry

end balls_in_boxes_l190_190415


namespace count_m_in_A_l190_190142

def A : Set ℕ := { 
  x | ∃ (a0 a1 a2 a3 : ℕ), a0 ∈ Finset.range 8 ∧ 
                           a1 ∈ Finset.range 8 ∧ 
                           a2 ∈ Finset.range 8 ∧ 
                           a3 ∈ Finset.range 8 ∧ 
                           a3 ≠ 0 ∧ 
                           x = a0 + a1 * 8 + a2 * 8^2 + a3 * 8^3 }

theorem count_m_in_A (m n : ℕ) (hA_m : m ∈ A) (hA_n : n ∈ A) (h_sum : m + n = 2018) (h_m_gt_n : m > n) :
  ∃! (count : ℕ), count = 497 := 
sorry

end count_m_in_A_l190_190142


namespace trapezoid_area_eq_c_l190_190292

theorem trapezoid_area_eq_c (b c : ℝ) (hb : b = Real.sqrt c) (hc : 0 < c) :
    let shorter_base := b - 3
    let altitude := b
    let longer_base := b + 3
    let K := (1/2) * (shorter_base + longer_base) * altitude
    K = c :=
by
    sorry

end trapezoid_area_eq_c_l190_190292


namespace polynomial_value_at_neg3_l190_190547

def polynomial (a b c x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 7

theorem polynomial_value_at_neg3 (a b c : ℝ) (h : polynomial a b c 3 = 65) :
  polynomial a b c (-3) = -79 := 
sorry

end polynomial_value_at_neg3_l190_190547


namespace number_of_adult_dogs_l190_190858

theorem number_of_adult_dogs (x : ℕ) (h : 2 * 50 + x * 100 + 2 * 150 = 700) : x = 3 :=
by
  -- Definitions from conditions
  have cost_cats := 2 * 50
  have cost_puppies := 2 * 150
  have total_cost := 700
  
  -- Using the provided hypothesis to assert our proof
  sorry

end number_of_adult_dogs_l190_190858


namespace LimingFatherAge_l190_190005

theorem LimingFatherAge
  (age month day : ℕ)
  (age_condition : 18 ≤ age ∧ age ≤ 70)
  (product_condition : age * month * day = 2975)
  (valid_month : 1 ≤ month ∧ month ≤ 12)
  (valid_day : 1 ≤ day ∧ day ≤ 31)
  : age = 35 := sorry

end LimingFatherAge_l190_190005


namespace sandy_puppies_l190_190121

theorem sandy_puppies :
  ∀ (initial_puppies puppies_given_away remaining_puppies : ℕ),
  initial_puppies = 8 →
  puppies_given_away = 4 →
  remaining_puppies = initial_puppies - puppies_given_away →
  remaining_puppies = 4 :=
by
  intros initial_puppies puppies_given_away remaining_puppies
  intro h_initial
  intro h_given_away
  intro h_remaining
  rw [h_initial, h_given_away] at h_remaining
  exact h_remaining

end sandy_puppies_l190_190121


namespace quadratic_has_two_distinct_real_roots_l190_190148

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2 * x1 + m = 0) ∧ (x2^2 - 2 * x2 + m = 0)) ↔ (m < 1) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l190_190148


namespace cos_double_angle_of_tan_half_l190_190662

theorem cos_double_angle_of_tan_half (α : ℝ) (h : Real.tan α = 1 / 2) :
  Real.cos (2 * α) = 3 / 5 :=
sorry

end cos_double_angle_of_tan_half_l190_190662


namespace find_common_ratio_l190_190409

-- Defining the conditions in Lean
variables (a : ℕ → ℝ) (d q : ℝ)

-- The arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) - a n = d

-- The geometric sequence condition
def is_geometric_sequence (a1 a2 a4 q : ℝ) : Prop :=
a2 ^ 2 = a1 * a4

-- Proving the main theorem
theorem find_common_ratio (a : ℕ → ℝ) (d q : ℝ) (h_arith : is_arithmetic_sequence a d) (d_ne_zero : d ≠ 0) 
(h_geom : is_geometric_sequence (a 1) (a 2) (a 4) q) : q = 2 :=
by
  sorry

end find_common_ratio_l190_190409


namespace smallest_positive_integer_l190_190967

-- We define the integers 3003 and 55555 as given in the conditions
def a : ℤ := 3003
def b : ℤ := 55555

-- The main theorem stating the smallest positive integer that can be written in the form ax + by is 1
theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, a * m + b * n = 1 :=
by
  -- We need not provide the proof steps here, just state it
  sorry

end smallest_positive_integer_l190_190967


namespace perpendicular_vectors_x_value_l190_190976

theorem perpendicular_vectors_x_value:
  ∀ (x : ℝ), let a : ℝ × ℝ := (1, 2)
             let b : ℝ × ℝ := (x, 1)
             (a.1 * b.1 + a.2 * b.2 = 0) → x = -2 :=
by
  intro x
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  intro h
  sorry

end perpendicular_vectors_x_value_l190_190976


namespace polynomial_use_square_of_binomial_form_l190_190663

theorem polynomial_use_square_of_binomial_form (a b x y : ℝ) :
  (1 + x) * (x + 1) = (x + 1) ^ 2 ∧ 
  (2 * a + b) * (b - 2 * a) = b^2 - 4 * a^2 ∧ 
  (-a + b) * (a - b) = - (a - b)^2 ∧ 
  (x^2 - y) * (y^2 + x) ≠ (x + y)^2 :=
by 
  sorry

end polynomial_use_square_of_binomial_form_l190_190663


namespace principal_amount_l190_190310

theorem principal_amount (P R : ℝ) (h1 : P + (P * R * 2) / 100 = 780) (h2 : P + (P * R * 7) / 100 = 1020) : P = 684 := 
sorry

end principal_amount_l190_190310


namespace no_real_solutions_if_discriminant_neg_one_real_solution_if_discriminant_zero_more_than_one_real_solution_if_discriminant_pos_l190_190851

noncomputable def system_discriminant (a b c : ℝ) : ℝ := (b - 1)^2 - 4 * a * c

theorem no_real_solutions_if_discriminant_neg (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c < 0) :
  ¬∃ (x₁ x₂ x₃ : ℝ), (a * x₁^2 + b * x₁ + c = x₂) ∧
                      (a * x₂^2 + b * x₂ + c = x₃) ∧
                      (a * x₃^2 + b * x₃ + c = x₁) :=
sorry

theorem one_real_solution_if_discriminant_zero (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c = 0) :
  ∃ (x : ℝ), ∀ (x₁ x₂ x₃ : ℝ), (x₁ = x) ∧ (x₂ = x) ∧ (x₃ = x) ∧
                              (a * x₁^2 + b * x₁ + c = x₂) ∧
                              (a * x₂^2 + b * x₂ + c = x₃) ∧
                              (a * x₃^2 + b * x₃ + c = x₁)  :=
sorry

theorem more_than_one_real_solution_if_discriminant_pos (a b c : ℝ) (h : a ≠ 0)
  (h_discriminant : (b - 1)^2 - 4 * a * c > 0) :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = x₂) ∧
                      (a * x₂^2 + b * x₂ + c = x₃) ∧
                      (a * x₃^2 + b * x₃ + c = x₁) :=
sorry

end no_real_solutions_if_discriminant_neg_one_real_solution_if_discriminant_zero_more_than_one_real_solution_if_discriminant_pos_l190_190851


namespace relationship_ab_l190_190927

noncomputable def a : ℝ := Real.log 243 / Real.log 5
noncomputable def b : ℝ := Real.log 27 / Real.log 3

theorem relationship_ab : a = (5 / 3) * b := sorry

end relationship_ab_l190_190927


namespace remainder_of_product_mod_5_l190_190806

theorem remainder_of_product_mod_5 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := by
  sorry

end remainder_of_product_mod_5_l190_190806


namespace cost_of_one_bag_of_onions_l190_190433

theorem cost_of_one_bag_of_onions (price_per_onion : ℕ) (total_onions : ℕ) (num_bags : ℕ) (h_price : price_per_onion = 200) (h_onions : total_onions = 180) (h_bags : num_bags = 6) :
  (total_onions / num_bags) * price_per_onion = 6000 := 
  by
  sorry

end cost_of_one_bag_of_onions_l190_190433


namespace sum_of_arithmetic_series_l190_190400

theorem sum_of_arithmetic_series (A B C : ℕ) (n : ℕ) 
  (hA : A = n * (2 * a₁ + (n - 1) * d) / 2)
  (hB : B = 2 * n * (2 * a₁ + (2 * n - 1) * d) / 2)
  (hC : C = 3 * n * (2 * a₁ + (3 * n - 1) * d) / 2) :
  C = 3 * (B - A) := sorry

end sum_of_arithmetic_series_l190_190400


namespace rabbit_roaming_area_l190_190601

noncomputable def rabbit_area_midpoint_long_side (r: ℝ) : ℝ :=
  (1/2) * Real.pi * r^2

noncomputable def rabbit_area_3_ft_from_corner (R r: ℝ) : ℝ :=
  (3/4) * Real.pi * R^2 - (1/4) * Real.pi * r^2

theorem rabbit_roaming_area (r R : ℝ) (h_r_pos: 0 < r) (h_R_pos: r < R) :
  rabbit_area_3_ft_from_corner R r - rabbit_area_midpoint_long_side R = 22.75 * Real.pi :=
by
  sorry

end rabbit_roaming_area_l190_190601


namespace f_eq_91_for_all_n_leq_100_l190_190036

noncomputable def f : ℤ → ℝ := sorry

theorem f_eq_91_for_all_n_leq_100 (n : ℤ) (h : n ≤ 100) : f n = 91 := sorry

end f_eq_91_for_all_n_leq_100_l190_190036


namespace derivative_at_one_l190_190656

noncomputable def f (x : ℝ) : ℝ := x / (x - 2)

theorem derivative_at_one : deriv f 1 = -2 :=
by 
  -- Here we would provide the proof that f'(1) = -2
  sorry

end derivative_at_one_l190_190656


namespace triangle_side_range_l190_190114

theorem triangle_side_range (a : ℝ) :
  1 < a ∧ a < 4 ↔ 3 + (2 * a - 1) > 4 ∧ 3 + 4 > 2 * a - 1 ∧ 4 + (2 * a - 1) > 3 :=
by
  sorry

end triangle_side_range_l190_190114


namespace sin_alpha_plus_beta_alpha_plus_two_beta_l190_190877

variables {α β : ℝ} (hα_acute : 0 < α ∧ α < π / 2) (hβ_acute : 0 < β ∧ β < π / 2)
          (h_tan_α : Real.tan α = 1 / 7) (h_sin_β : Real.sin β = Real.sqrt 10 / 10)

theorem sin_alpha_plus_beta : 
    Real.sin (α + β) = Real.sqrt 5 / 5 :=
by
  sorry

theorem alpha_plus_two_beta : 
    α + 2 * β = π / 4 :=
by
  sorry

end sin_alpha_plus_beta_alpha_plus_two_beta_l190_190877


namespace fraction_to_terminating_decimal_l190_190494

theorem fraction_to_terminating_decimal :
  (45 / (2^2 * 5^3) : ℚ) = 0.09 :=
by sorry

end fraction_to_terminating_decimal_l190_190494


namespace find_age_of_B_l190_190185

-- Define A and B as natural numbers (assuming ages are non-negative integers)
variables (A B : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := A + 10 = 2 * (B - 10)
def condition2 : Prop := A = B + 6

-- The goal is to prove that B = 36 given the conditions
theorem find_age_of_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 36 :=
sorry

end find_age_of_B_l190_190185


namespace average_age_combined_rooms_l190_190371

theorem average_age_combined_rooms :
  (8 * 30 + 5 * 22) / (8 + 5) = 26.9 := by
  sorry

end average_age_combined_rooms_l190_190371


namespace no_two_items_share_color_l190_190468

theorem no_two_items_share_color (shirts pants hats : Fin 5) :
  ∃ num_outfits : ℕ, num_outfits = 60 :=
by
  sorry

end no_two_items_share_color_l190_190468


namespace price_second_oil_per_litre_is_correct_l190_190519

-- Definitions based on conditions
def price_first_oil_per_litre := 54
def volume_first_oil := 10
def volume_second_oil := 5
def mixture_rate_per_litre := 58
def total_volume := volume_first_oil + volume_second_oil
def total_cost_mixture := total_volume * mixture_rate_per_litre
def total_cost_first_oil := volume_first_oil * price_first_oil_per_litre

-- The statement to prove
theorem price_second_oil_per_litre_is_correct (x : ℕ) (h : total_cost_first_oil + (volume_second_oil * x) = total_cost_mixture) : x = 66 :=
by
  sorry

end price_second_oil_per_litre_is_correct_l190_190519


namespace fermat_little_theorem_variant_l190_190014

theorem fermat_little_theorem_variant (p : ℕ) (m : ℤ) [hp : Fact (Nat.Prime p)] : 
  (m ^ p - m) % p = 0 :=
sorry

end fermat_little_theorem_variant_l190_190014


namespace business_fraction_l190_190500

theorem business_fraction (x : ℚ) (H1 : 3 / 4 * x * 60000 = 30000) : x = 2 / 3 :=
by sorry

end business_fraction_l190_190500


namespace quadratic_equation_in_x_l190_190128

theorem quadratic_equation_in_x (k x : ℝ) : 
  (k^2 + 1) * x^2 - (k * x - 8) - 1 = 0 := 
sorry

end quadratic_equation_in_x_l190_190128


namespace yellow_flower_count_l190_190404

-- Define the number of flowers of each color and total flowers based on given conditions
def total_flowers : Nat := 96
def green_flowers : Nat := 9
def red_flowers : Nat := 3 * green_flowers
def blue_flowers : Nat := total_flowers / 2

-- Define the number of yellow flowers
def yellow_flowers : Nat := total_flowers - (green_flowers + red_flowers + blue_flowers)

-- The theorem we aim to prove
theorem yellow_flower_count : yellow_flowers = 12 := by
  sorry

end yellow_flower_count_l190_190404


namespace red_light_max_probability_l190_190463

theorem red_light_max_probability {m : ℕ} (h1 : m > 0) (h2 : m < 35) :
  m = 3 ∨ m = 15 ∨ m = 30 ∨ m = 40 → m = 30 :=
by
  sorry

end red_light_max_probability_l190_190463


namespace instantaneous_velocity_at_2_l190_190064

def s (t : ℝ) : ℝ := 3 * t^2 + t

theorem instantaneous_velocity_at_2 : (deriv s 2) = 13 :=
by
  sorry

end instantaneous_velocity_at_2_l190_190064


namespace number_of_people_speaking_both_languages_l190_190181

theorem number_of_people_speaking_both_languages
  (total : ℕ) (L : ℕ) (F : ℕ) (N : ℕ) (B : ℕ) :
  total = 25 → L = 13 → F = 15 → N = 6 → total = L + F - B + N → B = 9 :=
by
  intros h_total h_L h_F h_N h_inclusion_exclusion
  sorry

end number_of_people_speaking_both_languages_l190_190181


namespace cost_of_pencil_pen_eraser_l190_190942

variables {p q r : ℝ}

theorem cost_of_pencil_pen_eraser 
  (h1 : 4 * p + 3 * q + r = 5.40)
  (h2 : 2 * p + 2 * q + 2 * r = 4.60) : 
  p + 2 * q + 3 * r = 4.60 := 
by sorry

end cost_of_pencil_pen_eraser_l190_190942


namespace description_of_T_l190_190309

-- Define the set T
def T : Set (ℝ × ℝ) := 
  {p | (p.1 = 1 ∧ p.2 ≤ 9) ∨ (p.2 = 9 ∧ p.1 ≤ 1) ∨ (p.2 = p.1 + 8 ∧ p.1 ≥ 1)}

-- State the formal proof problem: T is three rays with a common point
theorem description_of_T :
  (∃ p : ℝ × ℝ, p = (1, 9) ∧ 
    ∀ q ∈ T, 
      (q.1 = 1 ∧ q.2 ≤ 9) ∨ 
      (q.2 = 9 ∧ q.1 ≤ 1) ∨ 
      (q.2 = q.1 + 8 ∧ q.1 ≥ 1)) :=
by
  sorry

end description_of_T_l190_190309


namespace expression_eq_l190_190245

variable {α β γ δ p q : ℝ}

-- Conditions from the problem
def roots_eq1 (α β p : ℝ) : Prop := ∀ x : ℝ, (x - α) * (x - β) = x^2 + p*x - 1
def roots_eq2 (γ δ q : ℝ) : Prop := ∀ x : ℝ, (x - γ) * (x - δ) = x^2 + q*x + 1

-- The proof statement where the expression is equated to p^2 - q^2
theorem expression_eq (h1: roots_eq1 α β p) (h2: roots_eq2 γ δ q) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = p^2 - q^2 := sorry

end expression_eq_l190_190245


namespace range_of_m_l190_190188

theorem range_of_m (m : ℝ) :
  ( ∀ x : ℝ, |x + m| ≤ 4 → -2 ≤ x ∧ x ≤ 8) ↔ -4 ≤ m ∧ m ≤ -2 := 
by
  sorry

end range_of_m_l190_190188


namespace eval_expr_l190_190646

theorem eval_expr :
  - (18 / 3 * 8 - 48 + 4 * 6) = -24 := by
  sorry

end eval_expr_l190_190646


namespace mark_current_trees_l190_190123

theorem mark_current_trees (x : ℕ) (h : x + 12 = 25) : x = 13 :=
by {
  -- proof omitted
  sorry
}

end mark_current_trees_l190_190123


namespace calculate_n_l190_190084

theorem calculate_n (n : ℕ) : 3^n = 3 * 9^5 * 81^3 -> n = 23 :=
by
  -- Proof omitted
  sorry

end calculate_n_l190_190084


namespace complement_A_l190_190334

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := { x : ℝ | x < 2 }

theorem complement_A :
  (U \ A) = { x : ℝ | x >= 2 } :=
by
  sorry

end complement_A_l190_190334


namespace solve_system1_l190_190319

theorem solve_system1 (x y : ℝ) :
  x + y + 3 = 10 ∧ 4 * (x + y) - y = 25 →
  x = 4 ∧ y = 3 :=
by
  sorry

end solve_system1_l190_190319


namespace avery_work_time_l190_190925

theorem avery_work_time :
  ∀ (t : ℝ),
    (1/2 * t + 1/4 * 1 = 1) → t = 1 :=
by
  intros t h
  sorry

end avery_work_time_l190_190925


namespace negation_universal_proposition_l190_190555

theorem negation_universal_proposition :
  ¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 :=
by
  sorry

end negation_universal_proposition_l190_190555


namespace circle_radius_l190_190441

theorem circle_radius :
  ∃ radius : ℝ, (∀ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 16 → (x - 2)^2 + (y - 1)^2 = radius^2)
  ∧ radius = 4 :=
sorry

end circle_radius_l190_190441


namespace sunil_interest_l190_190912

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem sunil_interest :
  let A := 19828.80
  let r := 0.08
  let n := 1
  let t := 2
  let P := 19828.80 / (1 + 0.08) ^ 2
  P * (1 + r / n) ^ (n * t) = 19828.80 →
  A - P = 2828.80 :=
by
  sorry

end sunil_interest_l190_190912


namespace triangle_angle_C_l190_190611

theorem triangle_angle_C (A B C : ℝ) (h1 : A = 86) (h2 : B = 3 * C + 22) (h3 : A + B + C = 180) : C = 18 :=
by
  sorry

end triangle_angle_C_l190_190611


namespace circle_inequality_l190_190985

-- Given a circle of 100 pairwise distinct numbers a : ℕ → ℝ for 1 ≤ i ≤ 100
variables {a : ℕ → ℝ}
-- Hypothesis 1: distinct numbers
def distinct_numbers (a : ℕ → ℝ) := ∀ i j : ℕ, (1 ≤ i ∧ i ≤ 100) ∧ (1 ≤ j ∧ j ≤ 100) ∧ (i ≠ j) → a i ≠ a j

-- Theorem: Prove that there exist four consecutive numbers such that the sum of the first and the last number is strictly greater than the sum of the two middle numbers
theorem circle_inequality (h_distinct : distinct_numbers a) : 
  ∃ i : ℕ, (1 ≤ i ∧ i ≤ 100) ∧ (a i + a ((i + 3) % 100) > a ((i + 1) % 100) + a ((i + 2) % 100)) :=
sorry

end circle_inequality_l190_190985


namespace maximize_area_of_quadrilateral_l190_190264

theorem maximize_area_of_quadrilateral (k : ℝ) (h0 : 0 < k) (h1 : k < 1) 
    (hE : ∀ E : ℝ, E = 2 * k) (hF : ∀ F : ℝ, F = 2 * k) :
    k = 1/2 ∧ (2 * (1 - k) ^ 2) = 1/2 := 
by 
  sorry

end maximize_area_of_quadrilateral_l190_190264


namespace num_rectangular_arrays_with_48_chairs_l190_190569

theorem num_rectangular_arrays_with_48_chairs : 
  ∃ n, (∀ (r c : ℕ), 2 ≤ r ∧ 2 ≤ c ∧ r * c = 48 → (n = 8 ∨ n = 0)) ∧ (n = 8) :=
by 
  sorry

end num_rectangular_arrays_with_48_chairs_l190_190569


namespace car_value_reduction_l190_190911

/-- Jocelyn bought a car 3 years ago at $4000. 
If the car's value has reduced by 30%, calculate the current value of the car. 
Prove that it is equal to $2800. -/
theorem car_value_reduction (initial_value : ℝ) (reduction_percentage : ℝ) (current_value : ℝ) 
  (h_initial : initial_value = 4000)
  (h_reduction : reduction_percentage = 30)
  (h_current : current_value = initial_value - (reduction_percentage / 100) * initial_value) :
  current_value = 2800 :=
by
  -- Formal proof goes here
  sorry

end car_value_reduction_l190_190911


namespace stratified_sampling_l190_190091

noncomputable def employees := 500
noncomputable def under_35 := 125
noncomputable def between_35_and_49 := 280
noncomputable def over_50 := 95
noncomputable def sample_size := 100

theorem stratified_sampling : 
  under_35 * sample_size / employees = 25 := by
  sorry

end stratified_sampling_l190_190091


namespace cows_count_l190_190120

theorem cows_count (initial_cows last_year_deaths last_year_sales this_year_increase purchases gifts : ℕ)
  (h1 : initial_cows = 39)
  (h2 : last_year_deaths = 25)
  (h3 : last_year_sales = 6)
  (h4 : this_year_increase = 24)
  (h5 : purchases = 43)
  (h6 : gifts = 8) : 
  initial_cows - last_year_deaths - last_year_sales + this_year_increase + purchases + gifts = 83 := by
  sorry

end cows_count_l190_190120


namespace stock_price_calculation_l190_190417

def stock_price_end_of_first_year (initial_price : ℝ) (increase_percent : ℝ) : ℝ :=
  initial_price * (1 + increase_percent)

def stock_price_end_of_second_year (price_first_year : ℝ) (decrease_percent : ℝ) : ℝ :=
  price_first_year * (1 - decrease_percent)

theorem stock_price_calculation 
  (initial_price : ℝ)
  (increase_percent : ℝ)
  (decrease_percent : ℝ)
  (final_price : ℝ) :
  initial_price = 120 ∧ 
  increase_percent = 0.80 ∧
  decrease_percent = 0.30 ∧
  final_price = 151.20 → 
  stock_price_end_of_second_year (stock_price_end_of_first_year initial_price increase_percent) decrease_percent = final_price :=
by
  sorry

end stock_price_calculation_l190_190417


namespace solve_custom_operation_l190_190189

theorem solve_custom_operation (x : ℤ) (h : ((4 * 3 - (12 - x)) = 2)) : x = -2 :=
by
  sorry

end solve_custom_operation_l190_190189


namespace fishing_rod_price_l190_190138

theorem fishing_rod_price (initial_price : ℝ) 
  (price_increase_percentage : ℝ) 
  (price_decrease_percentage : ℝ) 
  (new_price : ℝ) 
  (final_price : ℝ) 
  (h1 : initial_price = 50) 
  (h2 : price_increase_percentage = 0.20) 
  (h3 : price_decrease_percentage = 0.15) 
  (h4 : new_price = initial_price * (1 + price_increase_percentage)) 
  (h5 : final_price = new_price * (1 - price_decrease_percentage)) 
  : final_price = 51 :=
sorry

end fishing_rod_price_l190_190138


namespace peter_pairs_of_pants_l190_190577

-- Define the conditions
def shirt_cost_condition (S : ℕ) : Prop := 2 * S = 20
def pants_cost (P : ℕ) : Prop := P = 6
def purchase_condition (P S : ℕ) (number_of_pants : ℕ) : Prop :=
  P * number_of_pants + 5 * S = 62

-- State the proof problem:
theorem peter_pairs_of_pants (S P number_of_pants : ℕ) 
  (h1 : shirt_cost_condition S)
  (h2 : pants_cost P) 
  (h3 : purchase_condition P S number_of_pants) :
  number_of_pants = 2 := by
  sorry

end peter_pairs_of_pants_l190_190577


namespace part1_l190_190372

def f (x : ℝ) := x^2 - 2*x

theorem part1 (x : ℝ) :
  (|f x| + |x^2 + 2*x| ≥ 6*|x|) ↔ (x ≤ -3 ∨ 3 ≤ x ∨ x = 0) :=
sorry

end part1_l190_190372


namespace probability_no_defective_pencils_l190_190311

theorem probability_no_defective_pencils :
  let total_pencils := 6
  let defective_pencils := 2
  let pencils_chosen := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_ways := Nat.choose total_pencils pencils_chosen
  let non_defective_ways := Nat.choose non_defective_pencils pencils_chosen
  (non_defective_ways / total_ways : ℚ) = 1 / 5 :=
by
  sorry

end probability_no_defective_pencils_l190_190311


namespace second_investment_amount_l190_190384

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

theorem second_investment_amount :
  ∀ (P₁ P₂ I₁ I₂ r t : ℝ), 
    P₁ = 5000 →
    I₁ = 250 →
    I₂ = 1000 →
    I₁ = simple_interest P₁ r t →
    I₂ = simple_interest P₂ r t →
    P₂ = 20000 := 
by 
  intros P₁ P₂ I₁ I₂ r t hP₁ hI₁ hI₂ hI₁_eq hI₂_eq
  sorry

end second_investment_amount_l190_190384


namespace belt_and_road_scientific_notation_l190_190378

theorem belt_and_road_scientific_notation : 
  4600000000 = 4.6 * 10^9 := 
by
  sorry

end belt_and_road_scientific_notation_l190_190378


namespace maximum_value_conditions_l190_190807

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem maximum_value_conditions (x_0 : ℝ) (h_max : ∀ x : ℝ, f x ≤ f x_0) :
    f x_0 = x_0 ∧ f x_0 < 1 / 2 :=
by
  sorry

end maximum_value_conditions_l190_190807


namespace solve_eq1_solve_eq2_l190_190972

theorem solve_eq1 (x : ℝ) : (12 * (x - 1) ^ 2 = 3) ↔ (x = 3/2 ∨ x = 1/2) := 
by sorry

theorem solve_eq2 (x : ℝ) : ((x + 1) ^ 3 = 0.125) ↔ (x = -0.5) := 
by sorry

end solve_eq1_solve_eq2_l190_190972


namespace new_rectangle_area_eq_a_squared_l190_190177

theorem new_rectangle_area_eq_a_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let d := Real.sqrt (a^2 + b^2)
  let base := 2 * (d + b)
  let height := (d - b) / 2
  base * height = a^2 := by
  sorry

end new_rectangle_area_eq_a_squared_l190_190177


namespace range_of_2a_plus_3b_l190_190436

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3) (h3 : 2 < a - b) (h4 : a - b < 4) :
  -9/2 < 2*a + 3*b ∧ 2*a + 3*b < 13/2 :=
  sorry

end range_of_2a_plus_3b_l190_190436


namespace min_value_of_expression_l190_190659

theorem min_value_of_expression (x y : ℤ) (h : 4 * x + 5 * y = 7) : ∃ k : ℤ, 
  5 * Int.natAbs (3 + 5 * k) - 3 * Int.natAbs (-1 - 4 * k) = 1 :=
sorry

end min_value_of_expression_l190_190659


namespace montoya_budget_l190_190279

def percentage_food (groceries: ℝ) (eating_out: ℝ) : ℝ :=
  groceries + eating_out

def percentage_transportation_rent_utilities (transportation: ℝ) (rent: ℝ) (utilities: ℝ) : ℝ :=
  transportation + rent + utilities

def total_percentage (food: ℝ) (transportation_rent_utilities: ℝ) : ℝ :=
  food + transportation_rent_utilities

theorem montoya_budget :
  ∀ (groceries : ℝ) (eating_out : ℝ) (transportation : ℝ) (rent : ℝ) (utilities : ℝ),
    groceries = 0.6 → eating_out = 0.2 → transportation = 0.1 → rent = 0.05 → utilities = 0.05 →
    total_percentage (percentage_food groceries eating_out) (percentage_transportation_rent_utilities transportation rent utilities) = 1 :=
by
sorry

end montoya_budget_l190_190279


namespace circle_line_distance_l190_190392

theorem circle_line_distance (c : ℝ) : 
  (∃ (P₁ P₂ P₃ : ℝ × ℝ), 
     (P₁ ≠ P₂ ∧ P₂ ≠ P₃ ∧ P₁ ≠ P₃) ∧
     ((P₁.1 - 2)^2 + (P₁.2 - 2)^2 = 18) ∧
     ((P₂.1 - 2)^2 + (P₂.2 - 2)^2 = 18) ∧
     ((P₃.1 - 2)^2 + (P₃.2 - 2)^2 = 18) ∧
     (abs (P₁.1 - P₁.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2) ∧
     (abs (P₂.1 - P₂.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2) ∧
     (abs (P₃.1 - P₃.2 + c) / Real.sqrt 2 = 2 * Real.sqrt 2)) ↔ 
  -2 ≤ c ∧ c ≤ 2 :=
sorry

end circle_line_distance_l190_190392


namespace problem_statement_l190_190612

variable {x a : Real}

theorem problem_statement (h1 : x < a) (h2 : a < 0) : x^2 > a * x ∧ a * x > a^2 := 
sorry

end problem_statement_l190_190612


namespace recommendation_plans_count_l190_190099

def num_male : ℕ := 3
def num_female : ℕ := 2
def num_recommendations : ℕ := 5

def num_spots_russian : ℕ := 2
def num_spots_japanese : ℕ := 2
def num_spots_spanish : ℕ := 1

def condition_russian (males : ℕ) : Prop := males > 0
def condition_japanese (males : ℕ) : Prop := males > 0

theorem recommendation_plans_count : 
  (∃ (males_r : ℕ) (males_j : ℕ), condition_russian males_r ∧ condition_japanese males_j ∧ 
  num_male - males_r - males_j >= 0 ∧ males_r + males_j ≤ num_male ∧ 
  num_female + (num_male - males_r - males_j) >= num_recommendations - (num_spots_russian + num_spots_japanese + num_spots_spanish)) →
  (∃ (x : ℕ), x = 24) := by
  sorry

end recommendation_plans_count_l190_190099


namespace range_of_k_l190_190106

theorem range_of_k 
  (x1 x2 y1 y2 k : ℝ)
  (h1 : y1 = 2 * x1 - k * x1 + 1)
  (h2 : y2 = 2 * x2 - k * x2 + 1)
  (h3 : x1 ≠ x2)
  (h4 : (x1 - x2) * (y1 - y2) < 0) : k > 2 := 
sorry

end range_of_k_l190_190106


namespace probability_of_2_red_1_black_l190_190687

theorem probability_of_2_red_1_black :
  let P_red := 4 / 7
  let P_black := 3 / 7 
  let prob_RRB := P_red * P_red * P_black 
  let prob_RBR := P_red * P_black * P_red 
  let prob_BRR := P_black * P_red * P_red 
  let total_prob := 3 * prob_RRB
  total_prob = 144 / 343 :=
by
  sorry

end probability_of_2_red_1_black_l190_190687


namespace find_y_l190_190481

-- Define the conditions (inversely proportional and sum condition)
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop := x * y = k
def sum_condition (x y : ℝ) : Prop := x + y = 50 ∧ x = 3 * y

-- Given these conditions, prove the value of y when x = -12
theorem find_y (k x y : ℝ)
  (h1 : inversely_proportional x y k)
  (h2 : sum_condition 37.5 12.5)
  (hx : x = -12) :
  y = -39.0625 :=
sorry

end find_y_l190_190481


namespace geese_count_l190_190828

theorem geese_count (initial : ℕ) (flown_away : ℕ) (left : ℕ) 
  (h₁ : initial = 51) (h₂ : flown_away = 28) : 
  left = initial - flown_away → left = 23 := 
by
  sorry

end geese_count_l190_190828


namespace inequality_proof_inequality_equality_conditions_l190_190891

theorem inequality_proof
  (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hxy1 : x1 * y1 - z1 ^ 2 > 0) (hxy2 : x2 * y2 - z2 ^ 2 > 0) :
  (x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2 ≤ (1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2)) :=
sorry

theorem inequality_equality_conditions
  (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hxy1 : x1 * y1 - z1 ^ 2 > 0) (hxy2 : x2 * y2 - z2 ^ 2 > 0) :
  ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2 = (1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2))
  ↔ (x1 = x2 ∧ y1 = y2 ∧ z1 = z2)) :=
sorry

end inequality_proof_inequality_equality_conditions_l190_190891


namespace hotdogs_per_hour_l190_190856

-- Define the necessary conditions
def price_per_hotdog : ℝ := 2
def total_hours : ℝ := 10
def total_sales : ℝ := 200

-- Prove that the number of hot dogs sold per hour equals 10
theorem hotdogs_per_hour : (total_sales / total_hours) / price_per_hotdog = 10 :=
by
  sorry

end hotdogs_per_hour_l190_190856


namespace p_necessary_not_sufficient_q_l190_190125

-- Define the conditions p and q
def p (a : ℝ) : Prop := a < 1
def q (a : ℝ) : Prop := 0 < a ∧ a < 1

-- State the necessary but not sufficient condition theorem
theorem p_necessary_not_sufficient_q (a : ℝ) : p a → q a → p a ∧ ¬∀ (a : ℝ), p a → q a :=
by
  sorry

end p_necessary_not_sufficient_q_l190_190125


namespace bicycle_has_four_wheels_l190_190418

-- Define the universe and properties of cars
axiom Car : Type
axiom Bicycle : Car
axiom has_four_wheels : Car → Prop
axiom all_cars_have_four_wheels : ∀ c : Car, has_four_wheels c

-- Define the theorem
theorem bicycle_has_four_wheels : has_four_wheels Bicycle :=
by
  sorry

end bicycle_has_four_wheels_l190_190418


namespace find_function_expression_l190_190358

variable (f : ℝ → ℝ)
variable (P : ℝ → ℝ → ℝ)

-- conditions
axiom a1 : f 1 = 1
axiom a2 : ∀ (x y : ℝ), f (x + y) = f x + f y + 2 * y * (x + y) + 1

-- proof statement
theorem find_function_expression (x : ℕ) (h : x ≠ 0) : f x = x^2 + 3*x - 3 := sorry

end find_function_expression_l190_190358


namespace travel_A_to_D_l190_190932

-- Definitions for the number of roads between each pair of cities
def roads_A_to_B : ℕ := 3
def roads_A_to_C : ℕ := 1
def roads_B_to_C : ℕ := 2
def roads_B_to_D : ℕ := 1
def roads_C_to_D : ℕ := 3

-- Theorem stating the total number of ways to travel from A to D visiting each city exactly once
theorem travel_A_to_D : roads_A_to_B * roads_B_to_C * roads_C_to_D + roads_A_to_C * roads_B_to_C * roads_B_to_D = 20 :=
by
  -- Formal proof goes here
  sorry

end travel_A_to_D_l190_190932


namespace sum_of_odds_square_l190_190201

theorem sum_of_odds_square (n : ℕ) (h : 0 < n) : (Finset.range n).sum (λ i => 2 * i + 1) = n ^ 2 :=
sorry

end sum_of_odds_square_l190_190201


namespace geometric_sequence_common_ratio_l190_190951

theorem geometric_sequence_common_ratio :
  (∃ q : ℝ, 1 + q + q^2 = 13 ∧ (q = 3 ∨ q = -4)) :=
by
  sorry

end geometric_sequence_common_ratio_l190_190951


namespace expand_expression_l190_190297

theorem expand_expression (x y z : ℝ) :
  (2 * x + 15) * (3 * y + 20 * z + 25) = 
  6 * x * y + 40 * x * z + 50 * x + 45 * y + 300 * z + 375 :=
by
  sorry

end expand_expression_l190_190297


namespace mr_yadav_yearly_savings_l190_190730

theorem mr_yadav_yearly_savings (S : ℕ) (h1 : S * 3 / 5 * 1 / 2 = 1584) : S * 3 / 5 * 1 / 2 * 12 = 19008 :=
  sorry

end mr_yadav_yearly_savings_l190_190730


namespace additional_charge_l190_190225

variable (charge_first : ℝ) -- The charge for the first 1/5 of a mile
variable (total_charge : ℝ) -- Total charge for an 8-mile ride
variable (distance : ℝ) -- Total distance of the ride

theorem additional_charge 
  (h1 : charge_first = 3.50) 
  (h2 : total_charge = 19.1) 
  (h3 : distance = 8) :
  ∃ x : ℝ, x = 0.40 :=
  sorry

end additional_charge_l190_190225


namespace multiplication_solution_l190_190278

theorem multiplication_solution 
  (x : ℤ) 
  (h : 72517 * x = 724807415) : 
  x = 9999 := 
sorry

end multiplication_solution_l190_190278


namespace kaleb_games_per_box_l190_190023

theorem kaleb_games_per_box (initial_games sold_games boxes remaining_games games_per_box : ℕ)
  (h1 : initial_games = 76)
  (h2 : sold_games = 46)
  (h3 : boxes = 6)
  (h4 : remaining_games = initial_games - sold_games)
  (h5 : games_per_box = remaining_games / boxes) :
  games_per_box = 5 :=
sorry

end kaleb_games_per_box_l190_190023


namespace inequality_for_positive_reals_l190_190668

theorem inequality_for_positive_reals (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) (k : ℕ) (h_k : 2 ≤ k) :
  (a^k / (a + b) + b^k / (b + c) + c^k / (c + a) ≥ 3 / 2) :=
by
  intros
  sorry

end inequality_for_positive_reals_l190_190668


namespace horse_revolutions_l190_190479

theorem horse_revolutions (r1 r2  : ℝ) (rev1 rev2 : ℕ)
  (h1 : r1 = 30) (h2 : rev1 = 20) (h3 : r2 = 10) : rev2 = 60 :=
by
  sorry

end horse_revolutions_l190_190479


namespace worker_overtime_hours_l190_190453

theorem worker_overtime_hours :
  ∃ (x y : ℕ), 60 * x + 90 * y = 3240 ∧ x + y = 50 ∧ y = 8 :=
by
  sorry

end worker_overtime_hours_l190_190453


namespace sharp_sharp_sharp_20_l190_190717

def sharp (N : ℝ) : ℝ := (0.5 * N)^2 + 1

theorem sharp_sharp_sharp_20 : sharp (sharp (sharp 20)) = 1627102.64 :=
by
  sorry

end sharp_sharp_sharp_20_l190_190717


namespace at_least_one_zero_l190_190866

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem at_least_one_zero (p q : ℝ) (h_zero : ∃ m : ℝ, f m p q = 0 ∧ f (f (f m p q) p q) p q = 0) :
  f 0 p q = 0 ∨ f 1 p q = 0 :=
sorry

end at_least_one_zero_l190_190866


namespace mac_total_loss_is_correct_l190_190414

def day_1_value : ℝ := 6 * 0.075 + 2 * 0.0075
def day_2_value : ℝ := 10 * 0.0045 + 5 * 0.0036
def day_3_value : ℝ := 4 * 0.10 + 1 * 0.011
def day_4_value : ℝ := 7 * 0.013 + 5 * 0.038
def day_5_value : ℝ := 3 * 0.5 + 2 * 0.0019
def day_6_value : ℝ := 12 * 0.0072 + 3 * 0.0013
def day_7_value : ℝ := 8 * 0.045 + 6 * 0.0089

def total_value : ℝ := day_1_value + day_2_value + day_3_value + day_4_value + day_5_value + day_6_value + day_7_value

def daily_loss (total_value: ℝ): ℝ := total_value - 0.25

def total_loss : ℝ := daily_loss day_1_value + daily_loss day_2_value + daily_loss day_3_value + daily_loss day_4_value + daily_loss day_5_value + daily_loss day_6_value + daily_loss day_7_value

theorem mac_total_loss_is_correct : total_loss = 2.1619 := 
by 
  simp [day_1_value, day_2_value, day_3_value, day_4_value, day_5_value, day_6_value, day_7_value, daily_loss, total_loss]
  sorry

end mac_total_loss_is_correct_l190_190414


namespace initial_plants_count_l190_190801

theorem initial_plants_count (p : ℕ) 
    (h1 : p - 20 > 0)
    (h2 : (p - 20) / 2 > 0)
    (h3 : ((p - 20) / 2) - 1 > 0)
    (h4 : ((p - 20) / 2) - 1 = 4) : 
    p = 30 :=
by
  sorry

end initial_plants_count_l190_190801


namespace onions_on_scale_l190_190258

theorem onions_on_scale (N : ℕ) (W_total : ℕ) (W_removed : ℕ) (avg_remaining : ℕ) (avg_removed : ℕ) :
  W_total = 7680 →
  W_removed = 5 * 206 →
  avg_remaining = 190 →
  avg_removed = 206 →
  N = 40 :=
by
  sorry

end onions_on_scale_l190_190258


namespace profit_percentage_calc_l190_190058

noncomputable def sale_price_incl_tax : ℝ := 616
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def cost_price : ℝ := 531.03
noncomputable def expected_profit_percentage : ℝ := 5.45

theorem profit_percentage_calc :
  let sale_price_before_tax := sale_price_incl_tax / (1 + sales_tax_rate)
  let profit := sale_price_before_tax - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = expected_profit_percentage :=
by
  sorry

end profit_percentage_calc_l190_190058


namespace scientific_notation_l190_190921

theorem scientific_notation :
  686530000 = 6.8653 * 10^8 :=
sorry

end scientific_notation_l190_190921


namespace percentage_increase_edge_length_l190_190958

theorem percentage_increase_edge_length (a a' : ℝ) (h : 6 * (a')^2 = 6 * a^2 + 1.25 * 6 * a^2) : a' = 1.5 * a :=
by sorry

end percentage_increase_edge_length_l190_190958


namespace badminton_players_l190_190134

theorem badminton_players (B T N Both Total: ℕ) 
  (h1: Total = 35)
  (h2: T = 18)
  (h3: N = 5)
  (h4: Both = 3)
  : B = 15 :=
by
  -- The proof block is intentionally left out.
  sorry

end badminton_players_l190_190134


namespace abc_zero_l190_190202

-- Define the given conditions as hypotheses
theorem abc_zero (a b c : ℚ) 
  (h1 : (a^2 + 1)^3 = b + 1)
  (h2 : (b^2 + 1)^3 = c + 1)
  (h3 : (c^2 + 1)^3 = a + 1) : 
  a = 0 ∧ b = 0 ∧ c = 0 := 
sorry

end abc_zero_l190_190202


namespace local_min_c_value_l190_190062

-- Definition of the function f(x) with its local minimum condition
def f (x c : ℝ) := x * (x - c)^2

-- Theorem stating that for the given function f(x) to have a local minimum at x = 1, the value of c must be 1
theorem local_min_c_value (c : ℝ) (h : ∀ ε > 0, f 1 ε < f c ε) : c = 1 := sorry

end local_min_c_value_l190_190062


namespace find_a_l190_190419

-- Given Conditions
def is_hyperbola (a : ℝ) : Prop := ∀ x y : ℝ, (x^2 / a) - (y^2 / 2) = 1
def is_asymptote (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = 2 * x

-- Question
theorem find_a (a : ℝ) (f : ℝ → ℝ) (hyp : is_hyperbola a) (asym : is_asymptote f) : a = 1 / 2 :=
sorry

end find_a_l190_190419


namespace sufficient_condition_implies_range_l190_190126

def setA : Set ℝ := {x | 1 ≤ x ∧ x < 3}

def setB (a : ℝ) : Set ℝ := {x | x^2 - a * x ≤ x - a}

theorem sufficient_condition_implies_range (a : ℝ) :
  (∀ x, x ∉ setA → x ∉ setB a) → (1 ≤ a ∧ a < 3) :=
by
  sorry

end sufficient_condition_implies_range_l190_190126


namespace factor_difference_of_squares_example_l190_190082

theorem factor_difference_of_squares_example :
    (m : ℝ) → (m ^ 2 - 4 = (m + 2) * (m - 2)) :=
by
    intro m
    sorry

end factor_difference_of_squares_example_l190_190082


namespace no_real_intersections_l190_190902

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l190_190902


namespace tattoo_ratio_l190_190776

theorem tattoo_ratio (a j k : ℕ) (ha : a = 23) (hj : j = 10) (rel : a = k * j + 3) : a / j = 23 / 10 :=
by {
  -- Insert proof here
  sorry
}

end tattoo_ratio_l190_190776


namespace triplet_D_sum_not_one_l190_190608

def triplet_sum_not_equal_to_one : Prop :=
  (1.2 + -0.2 + 0.0 ≠ 1)

theorem triplet_D_sum_not_one : triplet_sum_not_equal_to_one := 
  by
    sorry

end triplet_D_sum_not_one_l190_190608


namespace negation_implication_l190_190863

theorem negation_implication (a b c : ℝ) : 
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) :=
by 
  sorry

end negation_implication_l190_190863


namespace difference_between_numbers_l190_190742

open Int

theorem difference_between_numbers (A B : ℕ) 
  (h1 : A + B = 1812) 
  (h2 : A = 7 * B + 4) : 
  A - B = 1360 :=
by
  sorry

end difference_between_numbers_l190_190742


namespace john_adds_and_subtracts_l190_190428

theorem john_adds_and_subtracts :
  (41^2 = 40^2 + 81) ∧ (39^2 = 40^2 - 79) :=
by {
  sorry
}

end john_adds_and_subtracts_l190_190428


namespace rectangle_no_shaded_square_l190_190172

noncomputable def total_rectangles (cols : ℕ) : ℕ :=
  (cols + 1) * (cols + 1 - 1) / 2

noncomputable def shaded_rectangles (cols : ℕ) : ℕ :=
  cols + 1 - 1

noncomputable def probability_no_shaded (cols : ℕ) : ℚ :=
  let n := total_rectangles cols
  let m := shaded_rectangles cols
  1 - (m / n)

theorem rectangle_no_shaded_square :
  probability_no_shaded 2003 = 2002 / 2003 :=
by
  sorry

end rectangle_no_shaded_square_l190_190172


namespace largest_divisor_of_product_of_three_consecutive_odd_integers_l190_190115

theorem largest_divisor_of_product_of_three_consecutive_odd_integers :
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d = 3 ∧ ∀ m : ℕ, m ∣ ((2*n-1)*(2*n+1)*(2*n+3)) → m ≤ d :=
by
  sorry

end largest_divisor_of_product_of_three_consecutive_odd_integers_l190_190115


namespace log4_21_correct_l190_190447

noncomputable def log4_21 (a b : ℝ) (h1 : Real.log 3 = a * Real.log 2)
                                     (h2 : Real.log 2 = b * Real.log 7) : ℝ :=
  (a * b + 1) / (2 * b)

theorem log4_21_correct (a b : ℝ) (h1 : Real.log 3 = a * Real.log 2) 
                        (h2 : Real.log 2 = b * Real.log 7) : 
  log4_21 a b h1 h2 = (a * b + 1) / (2 * b) := 
sorry

end log4_21_correct_l190_190447


namespace ratio_is_five_ninths_l190_190996

-- Define the conditions
def total_profit : ℕ := 48000
def total_income : ℕ := 108000

-- Define the total spending based on conditions
def total_spending : ℕ := total_income - total_profit

-- Define the ratio of spending to income
def ratio_spending_to_income : ℚ := total_spending / total_income

-- The theorem we need to prove
theorem ratio_is_five_ninths : ratio_spending_to_income = 5 / 9 := 
  sorry

end ratio_is_five_ninths_l190_190996


namespace number_of_technicians_l190_190342

/-- 
In a workshop, the average salary of all the workers is Rs. 8000. 
The average salary of some technicians is Rs. 12000 and the average salary of the rest is Rs. 6000. 
The total number of workers in the workshop is 24.
Prove that there are 8 technicians in the workshop.
-/
theorem number_of_technicians 
  (total_workers : ℕ) 
  (avg_salary_all : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_rest : ℕ) 
  (num_technicians rest_workers : ℕ) 
  (h_total : total_workers = num_technicians + rest_workers)
  (h_avg_salary : (num_technicians * avg_salary_technicians + rest_workers * avg_salary_rest) = total_workers * avg_salary_all)
  (h1 : total_workers = 24)
  (h2 : avg_salary_all = 8000)
  (h3 : avg_salary_technicians = 12000)
  (h4 : avg_salary_rest = 6000) :
  num_technicians = 8 :=
by
  sorry

end number_of_technicians_l190_190342


namespace ratio_largest_smallest_root_geometric_progression_l190_190368

theorem ratio_largest_smallest_root_geometric_progression (a b c d : ℤ)
  (h_poly : a * x^3 + b * x^2 + c * x + d = 0) 
  (h_in_geo_prog : ∃ r1 r2 r3 q : ℝ, r1 < r2 ∧ r2 < r3 ∧ r1 * q = r2 ∧ r2 * q = r3 ∧ q ≠ 0) : 
  ∃ R : ℝ, R = 1 := 
by
  sorry

end ratio_largest_smallest_root_geometric_progression_l190_190368


namespace total_marbles_l190_190699

variable (r : ℝ) -- number of red marbles
variable (b g y : ℝ) -- number of blue, green, and yellow marbles

-- Conditions
axiom h1 : r = 1.3 * b
axiom h2 : g = 1.5 * r
axiom h3 : y = 0.8 * g

/-- Theorem: The total number of marbles in the collection is 4.47 times the number of red marbles -/
theorem total_marbles (r b g y : ℝ) (h1 : r = 1.3 * b) (h2 : g = 1.5 * r) (h3 : y = 0.8 * g) :
  b + r + g + y = 4.47 * r :=
sorry

end total_marbles_l190_190699


namespace remainder_19_pow_19_plus_19_mod_20_l190_190457

theorem remainder_19_pow_19_plus_19_mod_20 : (19 ^ 19 + 19) % 20 = 18 := 
by {
  sorry
}

end remainder_19_pow_19_plus_19_mod_20_l190_190457


namespace final_middle_pile_cards_l190_190679

-- Definitions based on conditions
def initial_cards_per_pile (n : ℕ) (h : n ≥ 2) := n

def left_pile_after_step_2 (n : ℕ) (h : n ≥ 2) := n - 2
def middle_pile_after_step_2 (n : ℕ) (h : n ≥ 2) := n + 2
def right_pile_after_step_2 (n : ℕ) (h : n ≥ 2) := n

def right_pile_after_step_3 (n : ℕ) (h : n ≥ 2) := n - 1
def middle_pile_after_step_3 (n : ℕ) (h : n ≥ 2) := n + 3

def left_pile_after_step_4 (n : ℕ) (h : n ≥ 2) := n
def middle_pile_after_step_4 (n : ℕ) (h : n ≥ 2) := (n + 3) - n

-- The proof problem to solve
theorem final_middle_pile_cards (n : ℕ) (h : n ≥ 2) : middle_pile_after_step_4 n h = 5 :=
sorry

end final_middle_pile_cards_l190_190679


namespace three_gorges_dam_capacity_scientific_notation_l190_190618

theorem three_gorges_dam_capacity_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (16780000 : ℝ) = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.678 ∧ n = 7 :=
by
  sorry

end three_gorges_dam_capacity_scientific_notation_l190_190618


namespace wheat_bread_served_l190_190535

noncomputable def total_bread_served : ℝ := 0.6
noncomputable def white_bread_served : ℝ := 0.4

theorem wheat_bread_served : total_bread_served - white_bread_served = 0.2 :=
by
  sorry

end wheat_bread_served_l190_190535


namespace trigonometric_identity_l190_190076

theorem trigonometric_identity 
  (α m : ℝ) 
  (h : Real.tan (α / 2) = m) :
  (1 - 2 * (Real.sin (α / 2))^2) / (1 + Real.sin α) = (1 - m) / (1 + m) :=
by
  sorry

end trigonometric_identity_l190_190076


namespace geom_series_eq_l190_190266

noncomputable def C (n : ℕ) := 256 * (1 - 1 / (4^n)) / (3 / 4)
noncomputable def D (n : ℕ) := 1024 * (1 - 1 / ((-2)^n)) / (3 / 2)

theorem geom_series_eq (n : ℕ) (h : n ≥ 1) : C n = D n ↔ n = 1 :=
by
  sorry

end geom_series_eq_l190_190266


namespace functional_equation_solution_l190_190244

noncomputable def f (t : ℝ) (x : ℝ) := (t * (x - t)) / (t + 1)

noncomputable def g (t : ℝ) (x : ℝ) := t * (x - t)

theorem functional_equation_solution (t : ℝ) (ht : t ≠ -1) :
  ∀ x y : ℝ, f t (x + g t y) = x * f t y - y * f t x + g t x :=
by
  intros x y
  let fx := f t
  let gx := g t
  sorry

end functional_equation_solution_l190_190244


namespace comics_in_box_l190_190498

def comics_per_comic := 25
def total_pages := 150
def existing_comics := 5

def torn_comics := total_pages / comics_per_comic
def total_comics := torn_comics + existing_comics

theorem comics_in_box : total_comics = 11 := by
  sorry

end comics_in_box_l190_190498


namespace fraction_of_profit_b_received_l190_190112

theorem fraction_of_profit_b_received (capital months_a_share months_b_share : ℝ) 
  (hA_contrib : capital * (1/4) * months_a_share = capital * (15/4))
  (hB_contrib : capital * (3/4) * months_b_share = capital * (30/4)) :
  (30/45) = (2/3) :=
by sorry

end fraction_of_profit_b_received_l190_190112


namespace equivalent_single_percentage_change_l190_190580

theorem equivalent_single_percentage_change :
  let original_price : ℝ := 250
  let num_items : ℕ := 400
  let first_increase : ℝ := 0.15
  let second_increase : ℝ := 0.20
  let discount : ℝ := -0.10
  let third_increase : ℝ := 0.25

  -- Calculations
  let price_after_first_increase := original_price * (1 + first_increase)
  let price_after_second_increase := price_after_first_increase * (1 + second_increase)
  let price_after_discount := price_after_second_increase * (1 + discount)
  let final_price := price_after_discount * (1 + third_increase)

  -- Calculate percentage change
  let percentage_change := ((final_price - original_price) / original_price) * 100

  percentage_change = 55.25 :=
by
  sorry

end equivalent_single_percentage_change_l190_190580


namespace quadrants_containing_points_l190_190780

theorem quadrants_containing_points (x y : ℝ) :
  (y > x + 1) → (y > 3 - 2 * x) → 
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end quadrants_containing_points_l190_190780


namespace find_number_l190_190963

theorem find_number (x : ℕ) (h : 5 + x = 20) : x = 15 :=
sorry

end find_number_l190_190963


namespace min_value_a_plus_b_l190_190094

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + b = 2 * a * b) : a + b ≥ 2 + Real.sqrt 3 :=
sorry

end min_value_a_plus_b_l190_190094


namespace time_to_reach_ship_l190_190497

/-- The scuba diver's descent problem -/

def rate_of_descent : ℕ := 35  -- in feet per minute
def depth_of_ship : ℕ := 3500  -- in feet

theorem time_to_reach_ship : depth_of_ship / rate_of_descent = 100 := by
  sorry

end time_to_reach_ship_l190_190497


namespace sin_C_of_arith_prog_angles_l190_190814

theorem sin_C_of_arith_prog_angles (A B C a b : ℝ) (h_abc : A + B + C = Real.pi)
  (h_arith_prog : 2 * B = A + C) (h_a : a = Real.sqrt 2) (h_b : b = Real.sqrt 3) :
  Real.sin C = (Real.sqrt 2 + Real.sqrt 6) / 4 :=
sorry

end sin_C_of_arith_prog_angles_l190_190814


namespace dice_sum_not_20_l190_190767

/-- Given that Louise rolls four standard six-sided dice (with faces numbered from 1 to 6)
    and the product of the numbers on the upper faces is 216, prove that it is not possible
    for the sum of the upper faces to be 20. -/
theorem dice_sum_not_20 (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) 
                        (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) 
                        (product : a * b * c * d = 216) : a + b + c + d ≠ 20 := 
by sorry

end dice_sum_not_20_l190_190767


namespace longer_leg_smallest_triangle_l190_190242

noncomputable def length_of_longer_leg_of_smallest_triangle (n : ℕ) (a : ℝ) : ℝ :=
  if n = 0 then a 
  else if n = 1 then (a / 2) * Real.sqrt 3
  else if n = 2 then ((a / 2) * Real.sqrt 3 / 2) * Real.sqrt 3
  else ((a / 2) * Real.sqrt 3 / 2 * Real.sqrt 3 / 2) * Real.sqrt 3

theorem longer_leg_smallest_triangle : 
  length_of_longer_leg_of_smallest_triangle 3 10 = 45 / 8 := 
sorry

end longer_leg_smallest_triangle_l190_190242


namespace zero_of_my_function_l190_190661

-- Define the function y = e^(2x) - 1
noncomputable def my_function (x : ℝ) : ℝ :=
  Real.exp (2 * x) - 1

-- Statement that the zero of the function is at x = 0
theorem zero_of_my_function : my_function 0 = 0 :=
by sorry

end zero_of_my_function_l190_190661


namespace breadth_of_rectangle_l190_190363

theorem breadth_of_rectangle (b l : ℝ) (h1 : l * b = 24 * b) (h2 : l - b = 10) : b = 14 :=
by
  sorry

end breadth_of_rectangle_l190_190363


namespace identity_eq_coefficients_l190_190386

theorem identity_eq_coefficients (a b c d : ℝ) :
  (∀ x : ℝ, a * x + b = c * x + d) ↔ (a = c ∧ b = d) :=
by
  sorry

end identity_eq_coefficients_l190_190386


namespace total_spent_on_toys_l190_190883

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59
def total_spent : ℝ := 12.30

theorem total_spent_on_toys : football_cost + marbles_cost = total_spent :=
by sorry

end total_spent_on_toys_l190_190883


namespace find_q_from_min_y_l190_190655

variables (a p q m : ℝ)
variable (a_nonzero : a ≠ 0)
variable (min_y : ∀ x : ℝ, a*x^2 + p*x + q ≥ m)

theorem find_q_from_min_y :
  q = m + p^2 / (4 * a) :=
sorry

end find_q_from_min_y_l190_190655


namespace water_inflow_rate_in_tank_A_l190_190980

-- Definitions from the conditions
def capacity := 20
def inflow_rate_B := 4
def extra_time_A := 5

-- Target variable
noncomputable def inflow_rate_A : ℕ :=
  let time_B := capacity / inflow_rate_B
  let time_A := time_B + extra_time_A
  capacity / time_A

-- Hypotheses
def tank_capacity : capacity = 20 := rfl
def tank_B_inflow : inflow_rate_B = 4 := rfl
def tank_A_extra_time : extra_time_A = 5 := rfl

-- Theorem statement
theorem water_inflow_rate_in_tank_A : inflow_rate_A = 2 := by
  -- Proof would go here
  sorry

end water_inflow_rate_in_tank_A_l190_190980


namespace TV_cost_exact_l190_190077

theorem TV_cost_exact (savings : ℝ) (fraction_furniture : ℝ) (fraction_tv : ℝ) (original_savings : ℝ) (tv_cost : ℝ) :
  savings = 880 →
  fraction_furniture = 3 / 4 →
  fraction_tv = 1 - fraction_furniture →
  tv_cost = fraction_tv * savings →
  tv_cost = 220 :=
by
  sorry

end TV_cost_exact_l190_190077


namespace bread_cost_l190_190168

theorem bread_cost (H C B : ℕ) (h₁ : H = 150) (h₂ : C = 200) (h₃ : H + B = C) : B = 50 :=
by
  sorry

end bread_cost_l190_190168


namespace Tony_change_l190_190205

structure Conditions where
  bucket_capacity : ℕ := 2
  sandbox_depth : ℕ := 2
  sandbox_width : ℕ := 4
  sandbox_length : ℕ := 5
  sand_weight_per_cubic_foot : ℕ := 3
  water_per_4_trips : ℕ := 3
  water_bottle_cost : ℕ := 2
  water_bottle_volume: ℕ := 15
  initial_money : ℕ := 10

theorem Tony_change (conds : Conditions) : 
  let volume_sandbox := conds.sandbox_depth * conds.sandbox_width * conds.sandbox_length
  let total_weight_sand := volume_sandbox * conds.sand_weight_per_cubic_foot
  let trips := total_weight_sand / conds.bucket_capacity
  let drinks := trips / 4
  let total_water := drinks * conds.water_per_4_trips
  let bottles_needed := total_water / conds.water_bottle_volume
  let total_cost := bottles_needed * conds.water_bottle_cost
  let change := conds.initial_money - total_cost
  change = 4 := by
  /- calculations corresponding to steps that are translated from the solution to show that the 
     change indeed is $4 -/
  sorry

end Tony_change_l190_190205


namespace f_value_l190_190893

noncomputable def f : ℝ → ℝ
| x => if x > 1 then 2^(x-1) else Real.tan (Real.pi * x / 3)

theorem f_value : f (1 / f 2) = Real.sqrt 3 / 3 := by
  sorry

end f_value_l190_190893


namespace ellipse_properties_l190_190353

theorem ellipse_properties
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b ≥ 0)
  (e : ℝ)
  (hc : e = 4 / 5)
  (directrix : ℝ)
  (hd : directrix = 25 / 4)
  (x y : ℝ)
  (hx : (x - 6)^2 / 25 + (y - 6)^2 / 9 = 1) :
  x^2 / 25 + y^2 / 9 = 1 :=
sorry

end ellipse_properties_l190_190353


namespace zoo_problem_l190_190948

theorem zoo_problem (M B L : ℕ) (h1: 26 ≤ M + B + L) (h2: M + B + L ≤ 32) 
    (h3: M + L > B) (h4: B + L = 2 * M) (h5: M + B = 3 * L + 3) (h6: B = L / 2) : 
    B = 3 :=
by
  sorry

end zoo_problem_l190_190948


namespace prime_ratio_sum_l190_190214

theorem prime_ratio_sum (p q m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
(h_roots : ∀ x : ℝ, x^2 - 99 * x + m = 0 → x = p ∨ x = q) :
  (p : ℚ) / q + q / p = 9413 / 194 :=
sorry

end prime_ratio_sum_l190_190214


namespace maize_storage_l190_190988

theorem maize_storage (x : ℝ)
  (h1 : 24 * x - 5 + 8 = 27) : x = 1 :=
  sorry

end maize_storage_l190_190988


namespace matrix_determinant_zero_implies_sum_of_squares_l190_190195

theorem matrix_determinant_zero_implies_sum_of_squares (a b : ℝ)
  (h : (Matrix.det ![![a - Complex.I, b - 2 * Complex.I],
                       ![1, 1 + Complex.I]]) = 0) :
  a^2 + b^2 = 1 :=
sorry

end matrix_determinant_zero_implies_sum_of_squares_l190_190195


namespace sin_2alpha_over_cos_alpha_sin_beta_value_l190_190947

variable (α β : ℝ)

-- Given conditions
axiom alpha_pos : 0 < α
axiom alpha_lt_pi_div_2 : α < Real.pi / 2
axiom beta_pos : 0 < β
axiom beta_lt_pi_div_2 : β < Real.pi / 2
axiom cos_alpha_eq : Real.cos α = 3 / 5
axiom cos_beta_plus_alpha_eq : Real.cos (β + α) = 5 / 13

-- The results to prove
theorem sin_2alpha_over_cos_alpha : (Real.sin (2 * α) / (Real.cos α ^ 2 + Real.cos (2 * α)) = 12) :=
sorry

theorem sin_beta_value : (Real.sin β = 16 / 65) :=
sorry


end sin_2alpha_over_cos_alpha_sin_beta_value_l190_190947


namespace builder_windows_installed_l190_190774

theorem builder_windows_installed (total_windows : ℕ) (hours_per_window : ℕ) (total_hours_left : ℕ) :
  total_windows = 14 → hours_per_window = 4 → total_hours_left = 36 → (total_windows - total_hours_left / hours_per_window) = 5 :=
by
  intros
  sorry

end builder_windows_installed_l190_190774


namespace range_of_m_l190_190256

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem range_of_m (m : ℝ) (h : second_quadrant (m-3) (m-2)) : 2 < m ∧ m < 3 :=
sorry

end range_of_m_l190_190256


namespace find_abcd_from_N_l190_190878

theorem find_abcd_from_N (N : ℕ) (hN1 : N ≥ 10000) (hN2 : N < 100000)
  (hN3 : N % 100000 = (N ^ 2) % 100000) : (N / 10) / 10 / 10 / 10 = 2999 := by
  sorry

end find_abcd_from_N_l190_190878


namespace DanteSoldCoconuts_l190_190466

variable (Paolo_coconuts : ℕ) (Dante_coconuts : ℕ) (coconuts_left : ℕ)

def PaoloHasCoconuts := Paolo_coconuts = 14

def DanteHasThriceCoconuts := Dante_coconuts = 3 * Paolo_coconuts

def DanteLeftCoconuts := coconuts_left = 32

theorem DanteSoldCoconuts 
  (h1 : PaoloHasCoconuts Paolo_coconuts) 
  (h2 : DanteHasThriceCoconuts Paolo_coconuts Dante_coconuts) 
  (h3 : DanteLeftCoconuts coconuts_left) : 
  Dante_coconuts - coconuts_left = 10 := 
by
  rw [PaoloHasCoconuts, DanteHasThriceCoconuts, DanteLeftCoconuts] at *
  sorry

end DanteSoldCoconuts_l190_190466


namespace geometric_sequence_common_ratio_l190_190634

theorem geometric_sequence_common_ratio (a₁ : ℕ) (S₃ : ℕ) (q : ℤ) 
  (h₁ : a₁ = 2) (h₂ : S₃ = 6) : 
  (q = 1 ∨ q = -2) :=
by
  sorry

end geometric_sequence_common_ratio_l190_190634


namespace area_bounded_by_parabola_and_x_axis_l190_190538

/-- Define the parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x - x^2

/-- The function for the x-axis -/
def x_axis : ℝ := 0

/-- Prove that the area bounded by the parabola and x-axis between x = 0 and x = 2 is 4/3 -/
theorem area_bounded_by_parabola_and_x_axis : 
  (∫ x in (0 : ℝ)..(2 : ℝ), parabola x) = 4 / 3 := by
    sorry

end area_bounded_by_parabola_and_x_axis_l190_190538


namespace new_average_doubled_marks_l190_190644

theorem new_average_doubled_marks (n : ℕ) (avg : ℕ) (h_n : n = 11) (h_avg : avg = 36) :
  (2 * avg * n) / n = 72 :=
by
  sorry

end new_average_doubled_marks_l190_190644


namespace sum_first_ten_multiples_of_nine_l190_190263

theorem sum_first_ten_multiples_of_nine :
  let a := 9
  let d := 9
  let n := 10
  let S_n := n * (2 * a + (n - 1) * d) / 2
  S_n = 495 := 
by
  sorry

end sum_first_ten_multiples_of_nine_l190_190263


namespace translate_parabola_l190_190670

noncomputable def f (x : ℝ) : ℝ := 3 * x^2

noncomputable def g (x : ℝ) : ℝ := 3 * (x - 1)^2 - 4

theorem translate_parabola (x : ℝ) : g x = 3 * (x - 1)^2 - 4 :=
by {
  -- proof would go here
  sorry
}

end translate_parabola_l190_190670


namespace sector_angle_solution_l190_190553

theorem sector_angle_solution (R α : ℝ) (h1 : 2 * R + α * R = 6) (h2 : (1/2) * R^2 * α = 2) : α = 1 ∨ α = 4 := 
sorry

end sector_angle_solution_l190_190553


namespace composite_for_large_n_l190_190514

theorem composite_for_large_n (m : ℕ) (hm : m > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → Nat.Prime (2^m * 2^(2^n) + 1) = false :=
sorry

end composite_for_large_n_l190_190514


namespace number_of_possible_tower_heights_l190_190698

-- Axiom for the possible increment values when switching brick orientations
def possible_increments : Set ℕ := {4, 7}

-- Base height when all bricks contribute the smallest dimension
def base_height (num_bricks : ℕ) (smallest_side : ℕ) : ℕ :=
  num_bricks * smallest_side

-- Check if a given height can be achieved by changing orientations of the bricks
def can_achieve_height (h : ℕ) (n : ℕ) (increments : Set ℕ) : Prop :=
  ∃ m k : ℕ, h = base_height n 2 + m * 4 + k * 7

-- Final proof statement
theorem number_of_possible_tower_heights :
  (50 : ℕ) = 50 →
  (∀ k : ℕ, (100 + k * 4 <= 450) → can_achieve_height (100 + k * 4) 50 possible_increments) →
  ∃ (num_possible_heights : ℕ), num_possible_heights = 90 :=
by
  sorry

end number_of_possible_tower_heights_l190_190698


namespace triangle_angle_zero_degrees_l190_190017

theorem triangle_angle_zero_degrees {a b c : ℝ} (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  ∃ (C : ℝ), C = 0 ∧ c = 0 :=
sorry

end triangle_angle_zero_degrees_l190_190017


namespace gcd_poly_l190_190594

theorem gcd_poly (k : ℕ) : Nat.gcd ((4500 * k)^2 + 11 * (4500 * k) + 40) (4500 * k + 8) = 3 := by
  sorry

end gcd_poly_l190_190594


namespace original_price_per_kg_of_salt_l190_190754

variable {P X : ℝ}

theorem original_price_per_kg_of_salt (h1 : 400 / (0.8 * P) = X + 10)
    (h2 : 400 / P = X) : P = 10 :=
by
  sorry

end original_price_per_kg_of_salt_l190_190754


namespace remainder_when_divided_by_30_l190_190962

theorem remainder_when_divided_by_30 (y : ℤ)
  (h1 : 4 + y ≡ 9 [ZMOD 8])
  (h2 : 6 + y ≡ 8 [ZMOD 27])
  (h3 : 8 + y ≡ 27 [ZMOD 125]) :
  y ≡ 4 [ZMOD 30] :=
sorry

end remainder_when_divided_by_30_l190_190962


namespace peach_cost_l190_190379

theorem peach_cost 
  (total_fruits : ℕ := 32)
  (total_cost : ℕ := 52)
  (plum_cost : ℕ := 2)
  (num_plums : ℕ := 20)
  (cost_peach : ℕ) :
  (total_cost - (num_plums * plum_cost)) = cost_peach * (total_fruits - num_plums) →
  cost_peach = 1 :=
by
  intro h
  sorry

end peach_cost_l190_190379


namespace percent_game_of_thrones_altered_l190_190529

def votes_game_of_thrones : ℕ := 10
def votes_twilight : ℕ := 12
def votes_art_of_deal : ℕ := 20

def altered_votes_art_of_deal : ℕ := votes_art_of_deal - (votes_art_of_deal * 80 / 100)
def altered_votes_twilight : ℕ := votes_twilight / 2
def total_altered_votes : ℕ := altered_votes_art_of_deal + altered_votes_twilight + votes_game_of_thrones

theorem percent_game_of_thrones_altered :
  ((votes_game_of_thrones * 100) / total_altered_votes) = 50 := by
  sorry

end percent_game_of_thrones_altered_l190_190529


namespace final_speed_is_zero_l190_190053

-- Define physical constants and conversion
def initial_speed_kmh : ℝ := 189
def initial_speed_ms : ℝ := initial_speed_kmh * 0.277778
def deceleration : ℝ := -0.5
def distance : ℝ := 4000

-- The goal is to prove the final speed is 0 m/s
theorem final_speed_is_zero (v_i : ℝ) (a : ℝ) (d : ℝ) (v_f : ℝ) 
  (hv_i : v_i = initial_speed_ms) 
  (ha : a = deceleration) 
  (hd : d = distance) 
  (h : v_f^2 = v_i^2 + 2 * a * d) : 
  v_f = 0 := 
by 
  sorry 

end final_speed_is_zero_l190_190053


namespace gcd_228_1995_l190_190681

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := 
by
  sorry

end gcd_228_1995_l190_190681


namespace minimum_value_l190_190561

theorem minimum_value (x : ℝ) (hx : x > 0) : 4 * x^2 + 1 / x^3 ≥ 5 ∧ (4 * x^2 + 1 / x^3 = 5 ↔ x = 1) :=
by {
  sorry
}

end minimum_value_l190_190561


namespace max_value_l190_190336

theorem max_value (x y : ℝ) : 
  (x + 3 * y + 4) / (Real.sqrt (x ^ 2 + y ^ 2 + 4)) ≤ Real.sqrt 26 :=
by
  -- Proof should be here
  sorry

end max_value_l190_190336


namespace system_of_equations_solution_l190_190567

theorem system_of_equations_solution (x y z : ℝ) :
  x^2 - y * z = -23 ∧ y^2 - z * x = -4 ∧ z^2 - x * y = 34 →
  (x = 5 ∧ y = 6 ∧ z = 8) ∨ (x = -5 ∧ y = -6 ∧ z = -8) :=
by
  sorry

end system_of_equations_solution_l190_190567


namespace neg_p_range_of_x_neg_q_sufficient_not_necessary_for_neg_p_l190_190377

def p (x : ℝ) : Prop := (x^2 - x - 2) ≤ 0
def q (x m : ℝ) : Prop := (x^2 - x - m^2 - m) ≤ 0

theorem neg_p_range_of_x (x : ℝ) : ¬ p x → x > 2 ∨ x < -1 :=
by
-- proof steps here
sorry

theorem neg_q_sufficient_not_necessary_for_neg_p (m : ℝ) : 
  (∀ x, ¬ q x m → ¬ p x) ∧ (∃ x, p x → ¬ q x m) → m > 1 ∨ m < -2 :=
by
-- proof steps here
sorry

end neg_p_range_of_x_neg_q_sufficient_not_necessary_for_neg_p_l190_190377


namespace similar_right_triangles_l190_190512

theorem similar_right_triangles (x c : ℕ) 
  (h1 : 12 * 6 = 9 * x) 
  (h2 : c^2 = x^2 + 6^2) :
  x = 8 ∧ c = 10 :=
by
  sorry

end similar_right_triangles_l190_190512


namespace abs_neg_one_fourth_l190_190338

theorem abs_neg_one_fourth : |(- (1 / 4))| = (1 / 4) :=
by
  sorry

end abs_neg_one_fourth_l190_190338


namespace geometric_sequence_third_term_l190_190524

theorem geometric_sequence_third_term (q : ℝ) (b1 : ℝ) (h1 : abs q < 1)
    (h2 : b1 / (1 - q) = 8 / 5) (h3 : b1 * q = -1 / 2) :
    b1 * q^2 / 2 = 1 / 8 := by
  sorry

end geometric_sequence_third_term_l190_190524


namespace smallest_three_digit_integer_solution_l190_190869

theorem smallest_three_digit_integer_solution :
  ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    (∃ a b c : ℕ,
      n = 100 * a + 10 * b + c ∧
      1 ≤ a ∧ a ≤ 9 ∧
      0 ≤ b ∧ b ≤ 9 ∧ 
      0 ≤ c ∧ c ≤ 9 ∧
      2 * n = 100 * c + 10 * b + a + 5) ∧ 
    n = 102 := by
{
  sorry
}

end smallest_three_digit_integer_solution_l190_190869


namespace axis_of_symmetry_l190_190873

theorem axis_of_symmetry (x : ℝ) (h : x = -Real.pi / 12) :
  ∃ k : ℤ, 2 * x - Real.pi / 3 = k * Real.pi + Real.pi / 2 :=
sorry

end axis_of_symmetry_l190_190873


namespace remainder_of_concatenated_numbers_l190_190435

def concatenatedNumbers : ℕ :=
  let digits := List.range (50) -- [0, 1, 2, ..., 49]
  digits.foldl (fun acc d => acc * 10 ^ (Nat.digits 10 d).length + d) 0

theorem remainder_of_concatenated_numbers :
  concatenatedNumbers % 50 = 49 :=
by
  sorry

end remainder_of_concatenated_numbers_l190_190435


namespace speed_of_stream_l190_190109

-- Conditions
variables (b s : ℝ)

-- Downstream and upstream conditions
def downstream_speed := 150 = (b + s) * 5
def upstream_speed := 75 = (b - s) * 7

-- Goal statement
theorem speed_of_stream (h1 : downstream_speed b s) (h2 : upstream_speed b s) : s = 135/14 :=
by sorry

end speed_of_stream_l190_190109


namespace padic_zeros_l190_190420

variable {p : ℕ} (hp : p > 1)
variable {a : ℕ} (hnz : a % p ≠ 0)

theorem padic_zeros (k : ℕ) (hk : k ≥ 1) :
  (a^(p^(k-1)*(p-1)) - 1) % (p^k) = 0 :=
sorry

end padic_zeros_l190_190420


namespace age_of_15th_student_l190_190426

theorem age_of_15th_student (avg_age_15 avg_age_3 avg_age_11 : ℕ) 
  (h_avg_15 : avg_age_15 = 15) 
  (h_avg_3 : avg_age_3 = 14) 
  (h_avg_11 : avg_age_11 = 16) : 
  ∃ x : ℕ, x = 7 := 
by
  sorry

end age_of_15th_student_l190_190426


namespace smallest_possible_value_of_N_l190_190122

noncomputable def smallest_N (N : ℕ) : Prop :=
  ∃ l m n : ℕ, l * m * n = N ∧ (l - 1) * (m - 1) * (n - 1) = 378

theorem smallest_possible_value_of_N : smallest_N 560 :=
  by {
    sorry
  }

end smallest_possible_value_of_N_l190_190122


namespace wire_cut_problem_l190_190361

variable (x : ℝ)

theorem wire_cut_problem 
  (h₁ : x + (5 / 2) * x = 49) : x = 14 :=
by
  sorry

end wire_cut_problem_l190_190361


namespace positive_multiples_of_4_with_units_digit_4_l190_190369

theorem positive_multiples_of_4_with_units_digit_4 (n : ℕ) : 
  ∃ n ≤ 15, ∀ m, m = 4 + 10 * (n - 1) → m < 150 ∧ m % 10 = 4 :=
by {
  sorry
}

end positive_multiples_of_4_with_units_digit_4_l190_190369


namespace find_first_number_l190_190542

theorem find_first_number
  (avg1 : (20 + 40 + 60) / 3 = 40)
  (avg2 : 40 - 4 = (x + 70 + 28) / 3)
  (sum_eq : x + 70 + 28 = 108) :
  x = 10 :=
by
  sorry

end find_first_number_l190_190542


namespace initial_ratio_milk_water_l190_190651

theorem initial_ratio_milk_water (M W : ℕ) 
  (h1 : M + W = 45) 
  (h2 : M = 3 * (W + 3)) 
  : M / W = 4 := 
sorry

end initial_ratio_milk_water_l190_190651


namespace gcd_eight_digit_repeating_four_digit_l190_190095

theorem gcd_eight_digit_repeating_four_digit :
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) →
  Nat.gcd (10001 * n) (10001 * m) = 10001) :=
by
  intros n hn m hm
  sorry

end gcd_eight_digit_repeating_four_digit_l190_190095


namespace problem_statement_l190_190901

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem problem_statement :
  (∀ x : ℝ, f (x) = 0 → x = - Real.pi / 6) ∧ (∀ x : ℝ, f (x) = 4 * Real.cos (2 * x - Real.pi / 6)) := sorry

end problem_statement_l190_190901


namespace evaluation_of_expression_l190_190939

theorem evaluation_of_expression : (3^2 - 2^2 + 1^2) = 6 :=
by
  sorry

end evaluation_of_expression_l190_190939


namespace fraction_of_b_eq_three_tenths_a_l190_190862

theorem fraction_of_b_eq_three_tenths_a (a b : ℝ) (h1 : a + b = 100) (h2 : b = 60) :
  (3 / 10) * a = (1 / 5) * b :=
by 
  have h3 : a = 40 := by linarith [h1, h2]
  rw [h2, h3]
  linarith

end fraction_of_b_eq_three_tenths_a_l190_190862


namespace pentadecagon_diagonals_l190_190413

def numberOfDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem pentadecagon_diagonals : numberOfDiagonals 15 = 90 :=
by
  sorry

end pentadecagon_diagonals_l190_190413


namespace josie_gift_money_l190_190585

-- Define the cost of each cassette tape
def tape_cost : ℕ := 9

-- Define the number of cassette tapes Josie plans to buy
def num_tapes : ℕ := 2

-- Define the cost of the headphone set
def headphone_cost : ℕ := 25

-- Define the amount of money Josie will have left after the purchases
def money_left : ℕ := 7

-- Define the total cost of tapes
def total_tape_cost := num_tapes * tape_cost

-- Define the total cost of both tapes and headphone set
def total_cost := total_tape_cost + headphone_cost

-- The total money Josie will have would be total_cost + money_left
theorem josie_gift_money : total_cost + money_left = 50 :=
by
  -- Proof will be provided here
  sorry

end josie_gift_money_l190_190585


namespace probability_diff_colors_l190_190317

/-!
There are 5 identical balls, including 3 white balls and 2 black balls. 
If 2 balls are drawn at once, the probability of the event "the 2 balls have different colors" 
occurring is \( \frac{3}{5} \).
-/

theorem probability_diff_colors 
    (white_balls : ℕ) (black_balls : ℕ) (total_balls : ℕ) (drawn_balls : ℕ) 
    (h_white : white_balls = 3) (h_black : black_balls = 2) (h_total : total_balls = 5) (h_drawn : drawn_balls = 2) :
    let total_ways := Nat.choose total_balls drawn_balls
    let diff_color_ways := (Nat.choose white_balls 1) * (Nat.choose black_balls 1)
    (diff_color_ways : ℚ) / (total_ways : ℚ) = 3 / 5 := 
by
    -- Step 1: Calculate total ways to draw 2 balls out of 5
    -- total_ways = 10 (by binomial coefficient)
    -- Step 2: Calculate favorable outcomes (1 white, 1 black)
    -- diff_color_ways = 6
    -- Step 3: Calculate probability
    -- Probability = 6 / 10 = 3 / 5
    sorry

end probability_diff_colors_l190_190317


namespace Vanya_number_thought_of_l190_190160

theorem Vanya_number_thought_of :
  ∃ m n : ℕ, m < 10 ∧ n < 10 ∧ (10 * m + n = 81 ∧ (10 * n + m)^2 = 4 * (10 * m + n)) :=
sorry

end Vanya_number_thought_of_l190_190160


namespace find_income_l190_190544

variable (x : ℝ)

def income : ℝ := 5 * x
def expenditure : ℝ := 4 * x
def savings : ℝ := income x - expenditure x

theorem find_income (h : savings x = 4000) : income x = 20000 :=
by
  rw [savings, income, expenditure] at h
  sorry

end find_income_l190_190544


namespace number_that_multiplies_b_l190_190979

variable (a b x : ℝ)

theorem number_that_multiplies_b (h1 : 7 * a = x * b) (h2 : a * b ≠ 0) (h3 : (a / 8) / (b / 7) = 1) : x = 8 := 
sorry

end number_that_multiplies_b_l190_190979


namespace solution_set_of_inequality_l190_190957

theorem solution_set_of_inequality (x : ℝ) :
  (x - 1) * (x - 2) ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end solution_set_of_inequality_l190_190957


namespace percentage_of_students_chose_spring_is_10_l190_190971

-- Define the constants given in the problem
def total_students : ℕ := 10
def students_spring : ℕ := 1

-- Define the percentage calculation formula
def percentage (part total : ℕ) : ℕ := (part * 100) / total

-- State the theorem
theorem percentage_of_students_chose_spring_is_10 :
  percentage students_spring total_students = 10 :=
by
  -- We don't need to provide a proof here, just state it.
  sorry

end percentage_of_students_chose_spring_is_10_l190_190971


namespace minimum_value_fraction_l190_190090

theorem minimum_value_fraction (m n : ℝ) (h0 : 0 ≤ m) (h1 : 0 ≤ n) (h2 : m + n = 1) :
  ∃ min_val, min_val = (1 / 4) ∧ (∀ m n, 0 ≤ m → 0 ≤ n → m + n = 1 → (m^2) / (m + 2) + (n^2) / (n + 1) ≥ min_val) :=
sorry

end minimum_value_fraction_l190_190090


namespace mod_inverse_sum_l190_190233

theorem mod_inverse_sum :
  ∃ a b : ℕ, (5 * a ≡ 1 [MOD 21]) ∧ (b = (a * a) % 21) ∧ ((a + b) % 21 = 9) :=
by
  sorry

end mod_inverse_sum_l190_190233


namespace equal_sets_implies_value_of_m_l190_190041

theorem equal_sets_implies_value_of_m (m : ℝ) (A B : Set ℝ) (hA : A = {3, m}) (hB : B = {3 * m, 3}) (hAB : A = B) : m = 0 :=
by
  -- Proof goes here
  sorry

end equal_sets_implies_value_of_m_l190_190041


namespace seq_positive_integers_seq_not_divisible_by_2109_l190_190034

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ a 2 = 6 ∧ ∀ n : ℕ, a (n + 2) = (a (n + 1) ^ 2 + 9) / a n

theorem seq_positive_integers (a : ℕ → ℤ) (h : seq a) : ∀ n : ℕ, 0 < a (n + 1) :=
sorry

theorem seq_not_divisible_by_2109 (a : ℕ → ℤ) (h : seq a) : ¬ ∃ m : ℕ, 2109 ∣ a (m + 1) :=
sorry

end seq_positive_integers_seq_not_divisible_by_2109_l190_190034


namespace adam_age_l190_190868

theorem adam_age (x : ℤ) :
  (∃ m : ℤ, x - 2 = m^2) ∧ (∃ n : ℤ, x + 2 = n^3) → x = 6 :=
by
  sorry

end adam_age_l190_190868


namespace range_of_k_intersecting_hyperbola_l190_190629

theorem range_of_k_intersecting_hyperbola :
  (∀ b : ℝ, ∃ x y : ℝ, y = k * x + b ∧ x^2 - 2 * y^2 = 1) →
  -Real.sqrt 2 / 2 < k ∧ k < Real.sqrt 2 / 2 :=
sorry

end range_of_k_intersecting_hyperbola_l190_190629


namespace arithmetic_sequence_max_n_pos_sum_l190_190290

noncomputable def max_n (a : ℕ → ℤ) (d : ℤ) : ℕ :=
  8

theorem arithmetic_sequence_max_n_pos_sum
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith_seq : ∀ n, a (n+1) = a 1 + n * d)
  (h_a1 : a 1 > 0)
  (h_a4_a5_sum_pos : a 4 + a 5 > 0)
  (h_a4_a5_prod_neg : a 4 * a 5 < 0) :
  max_n a d = 8 := by
  sorry

end arithmetic_sequence_max_n_pos_sum_l190_190290


namespace fourth_sphere_radius_l190_190455

theorem fourth_sphere_radius (R r : ℝ) (h1 : R > 0)
  (h2 : ∀ (a b c d : ℝ × ℝ × ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a →
    dist a b = 2*R ∧ dist b c = 2*R ∧ dist c d = 2*R ∧ dist d a = R + r ∧
    dist a c = R + r ∧ dist b d = R + r) :
  r = 4*R/3 :=
  sorry

end fourth_sphere_radius_l190_190455


namespace no_consecutive_beeches_probability_l190_190398

theorem no_consecutive_beeches_probability :
  let total_trees := 12
  let oaks := 3
  let holm_oaks := 4
  let beeches := 5
  let total_arrangements := (Nat.factorial total_trees) / ((Nat.factorial oaks) * (Nat.factorial holm_oaks) * (Nat.factorial beeches))
  let favorable_arrangements :=
    let slots := oaks + holm_oaks + 1
    Nat.choose slots beeches * ((Nat.factorial (oaks + holm_oaks)) / ((Nat.factorial oaks) * (Nat.factorial holm_oaks)))
  let probability := favorable_arrangements / total_arrangements
  probability = 7 / 99 :=
by
  sorry

end no_consecutive_beeches_probability_l190_190398


namespace complement_intersection_l190_190771

open Set

variable (U : Set ℕ) (A B : Set ℕ)

theorem complement_intersection :
  U = {1, 2, 3, 4, 5} →
  A = {1, 2, 3} →
  B = {2, 3, 5} →
  U \ (A ∩ B) = {1, 4, 5} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  sorry

end complement_intersection_l190_190771


namespace airplane_rows_l190_190161

theorem airplane_rows (R : ℕ) 
  (h1 : ∀ n, n = 5) 
  (h2 : ∀ s, s = 7) 
  (h3 : ∀ f, f = 2) 
  (h4 : ∀ p, p = 1400):
  (2 * 5 * 7 * R = 1400) → R = 20 :=
by
  -- Assuming the given equation 2 * 5 * 7 * R = 1400
  sorry

end airplane_rows_l190_190161


namespace most_reasonable_sampling_method_l190_190632

-- Definitions based on the conditions in the problem:
def area_divided_into_200_plots : Prop := true
def plan_randomly_select_20_plots : Prop := true
def large_difference_in_plant_coverage : Prop := true
def goal_representative_sample_accurate_estimate : Prop := true

-- Main theorem statement
theorem most_reasonable_sampling_method
  (h1 : area_divided_into_200_plots)
  (h2 : plan_randomly_select_20_plots)
  (h3 : large_difference_in_plant_coverage)
  (h4 : goal_representative_sample_accurate_estimate) :
  Stratified_sampling := 
sorry

end most_reasonable_sampling_method_l190_190632
