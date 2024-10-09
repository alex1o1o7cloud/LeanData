import Mathlib

namespace intersection_points_vertex_of_function_value_of_m_shift_l843_84307

noncomputable def quadratic_function (x m : ℝ) : ℝ :=
  (x - m) ^ 2 - 2 * (x - m)

theorem intersection_points (m : ℝ) : 
  ∃ x, quadratic_function x m = 0 ↔ x = m ∨ x = m + 2 := 
by
  sorry

theorem vertex_of_function (m : ℝ) : 
  ∃ x y, y = quadratic_function x m 
  ∧ x = m + 1 ∧ y = -1 := 
by
  sorry

theorem value_of_m_shift (m : ℝ) :
  (m - 2 = 0) → m = 2 :=
by
  sorry

end intersection_points_vertex_of_function_value_of_m_shift_l843_84307


namespace profit_calculation_l843_84344

-- Define the initial conditions
def initial_cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def price_decrease_effect : ℝ := 4
def daily_profit_target : ℝ := 13600
def minimum_selling_price : ℝ := 150

-- Define the function relationship of daily sales volume with respect to x
def sales_volume (x : ℝ) : ℝ := initial_sales_volume + price_decrease_effect * x

-- Define the selling price
def selling_price (x : ℝ) : ℝ := initial_selling_price - x

-- Define the profit function
def profit (x : ℝ) : ℝ := (selling_price x - initial_cost_price) * sales_volume x

theorem profit_calculation (x : ℝ) (hx : selling_price x ≥ minimum_selling_price) :
  profit x = daily_profit_target ↔ selling_price x = 185 := by
  sorry

end profit_calculation_l843_84344


namespace last_three_digits_of_8_pow_105_l843_84356

theorem last_three_digits_of_8_pow_105 : (8 ^ 105) % 1000 = 992 :=
by
  sorry

end last_three_digits_of_8_pow_105_l843_84356


namespace find_x_value_l843_84366

noncomputable def log (a b: ℝ): ℝ := Real.log a / Real.log b

theorem find_x_value (a n : ℝ) (t y: ℝ):
  1 < a →
  1 < t →
  y = 8 →
  log n (a^t) - 3 * log a (a^t) * log y 8 = 3 →
  x = a^t →
  x = a^2 :=
by
  sorry

end find_x_value_l843_84366


namespace unique_subset_empty_set_l843_84369

def discriminant (a : ℝ) : ℝ := 4 - 4 * a^2

theorem unique_subset_empty_set (a : ℝ) :
  (∀ (x : ℝ), ¬(a * x^2 + 2 * x + a = 0)) ↔ (a > 1 ∨ a < -1) :=
by
  sorry

end unique_subset_empty_set_l843_84369


namespace functional_expression_selling_price_for_profit_l843_84370

-- Define the initial conditions
def cost_price : ℚ := 8
def initial_selling_price : ℚ := 10
def initial_sales_volume : ℚ := 200
def sales_decrement_per_yuan_increase : ℚ := 20

-- Functional expression between y (items) and x (yuan)
theorem functional_expression (x : ℚ) : 
  (200 - 20 * (x - 10) = -20 * x + 400) :=
sorry

-- Determine the selling price to achieve a daily profit of 640 yuan
theorem selling_price_for_profit (x : ℚ) (h1 : 8 ≤ x) (h2 : x ≤ 15) : 
  ((x - 8) * (400 - 20 * x) = 640) → (x = 12) :=
sorry

end functional_expression_selling_price_for_profit_l843_84370


namespace jonah_raisins_l843_84382

variable (y : ℝ)

theorem jonah_raisins :
  (y + 0.4 = 0.7) → (y = 0.3) :=
  by
  intro h
  sorry

end jonah_raisins_l843_84382


namespace find_z_given_conditions_l843_84336

variable (x y z : ℤ)

theorem find_z_given_conditions :
  (x + y) / 2 = 4 →
  x + y + z = 0 →
  z = -8 := by
  sorry

end find_z_given_conditions_l843_84336


namespace units_digit_5_pow_2023_l843_84327

theorem units_digit_5_pow_2023 : ∀ n : ℕ, (n > 0) → (5^n % 10 = 5) → (5^2023 % 10 = 5) :=
by
  intros n hn hu
  have h_units_digit : ∀ k : ℕ, (k > 0) → 5^k % 10 = 5 := by
    intro k hk
    sorry -- pattern proof not included
  exact h_units_digit 2023 (by norm_num)

end units_digit_5_pow_2023_l843_84327


namespace most_stable_machine_l843_84384

noncomputable def var_A : ℝ := 10.3
noncomputable def var_B : ℝ := 6.9
noncomputable def var_C : ℝ := 3.5

theorem most_stable_machine :
  (var_C < var_B) ∧ (var_C < var_A) :=
by
  sorry

end most_stable_machine_l843_84384


namespace veranda_area_l843_84326

theorem veranda_area (length_room width_room width_veranda : ℕ)
  (h_length : length_room = 20) 
  (h_width : width_room = 12) 
  (h_veranda : width_veranda = 2) : 
  (length_room + 2 * width_veranda) * (width_room + 2 * width_veranda) - (length_room * width_room) = 144 := 
by
  sorry

end veranda_area_l843_84326


namespace farthest_vertex_coordinates_l843_84373

noncomputable def image_vertex_coordinates_farthest_from_origin 
    (center_EFGH : ℝ × ℝ) (area_EFGH : ℝ) (dilation_center : ℝ × ℝ) 
    (scale_factor : ℝ) : ℝ × ℝ := sorry

theorem farthest_vertex_coordinates 
    (center_EFGH : ℝ × ℝ := (10, -6)) (area_EFGH : ℝ := 16) 
    (dilation_center : ℝ × ℝ := (2, 2)) (scale_factor : ℝ := 3) : 
    image_vertex_coordinates_farthest_from_origin center_EFGH area_EFGH dilation_center scale_factor = (32, -28) := 
sorry

end farthest_vertex_coordinates_l843_84373


namespace contrapositive_statement_l843_84311

-- Conditions: x and y are real numbers
variables (x y : ℝ)

-- Contrapositive statement: If x ≠ 0 or y ≠ 0, then x^2 + y^2 ≠ 0
theorem contrapositive_statement (hx : x ≠ 0 ∨ y ≠ 0) : x^2 + y^2 ≠ 0 :=
sorry

end contrapositive_statement_l843_84311


namespace sum_of_reciprocals_l843_84309

open Real

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x + y = 5 * x * y) (hx2y : x = 2 * y) : 
  (1 / x) + (1 / y) = 5 := 
  sorry

end sum_of_reciprocals_l843_84309


namespace Buffy_whiskers_l843_84389

def whiskers_Juniper : ℕ := 12
def whiskers_Puffy : ℕ := 3 * whiskers_Juniper
def whiskers_Scruffy : ℕ := 2 * whiskers_Puffy
def whiskers_Buffy : ℕ := (whiskers_Puffy + whiskers_Scruffy + whiskers_Juniper) / 3

theorem Buffy_whiskers : whiskers_Buffy = 40 := by
  sorry

end Buffy_whiskers_l843_84389


namespace number_of_intersections_l843_84322

-- Conditions for the problem
def Line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 2
def Line2 (x y : ℝ) : Prop := 5 * x + 3 * y = 6
def Line3 (x y : ℝ) : Prop := x - 4 * y = 8

-- Statement to prove
theorem number_of_intersections : ∃ (p1 p2 p3 : ℝ × ℝ), 
  (Line1 p1.1 p1.2 ∧ Line2 p1.1 p1.2) ∧ 
  (Line1 p2.1 p2.2 ∧ Line3 p2.1 p2.2) ∧ 
  (Line2 p3.1 p3.2 ∧ Line3 p3.1 p3.2) ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 :=
sorry

end number_of_intersections_l843_84322


namespace dot_but_not_straight_line_l843_84360

theorem dot_but_not_straight_line :
  let total := 80
  let D_n_S := 28
  let S_n_D := 47
  ∃ (D : ℕ), D - D_n_S = 5 ∧ D + S_n_D = total :=
by
  sorry

end dot_but_not_straight_line_l843_84360


namespace determine_f_4_l843_84308

theorem determine_f_4 (f g : ℝ → ℝ)
  (h1 : ∀ x y z : ℝ, f (x^2 + y * f z) = x * g x + z * g y)
  (h2 : ∀ x : ℝ, g x = 2 * x) :
  f 4 = 32 :=
sorry

end determine_f_4_l843_84308


namespace add_fractions_l843_84396

theorem add_fractions : (1 / 4 : ℚ) + (3 / 5) = 17 / 20 := 
by
  sorry

end add_fractions_l843_84396


namespace mod_product_eq_15_l843_84318

theorem mod_product_eq_15 :
  (15 * 24 * 14) % 25 = 15 :=
by
  sorry

end mod_product_eq_15_l843_84318


namespace skill_position_players_waiting_l843_84362

def linemen_drink : ℕ := 8
def skill_position_player_drink : ℕ := 6
def num_linemen : ℕ := 12
def num_skill_position_players : ℕ := 10
def cooler_capacity : ℕ := 126

theorem skill_position_players_waiting :
  num_skill_position_players - (cooler_capacity - num_linemen * linemen_drink) / skill_position_player_drink = 5 :=
by
  -- Calculation is needed to be filled in here
  sorry

end skill_position_players_waiting_l843_84362


namespace simplify_expr_l843_84332

def A (a b : ℝ) := b^2 - a^2 + 5 * a * b
def B (a b : ℝ) := 3 * a * b + 2 * b^2 - a^2

theorem simplify_expr (a b : ℝ) : 2 * (A a b) - (B a b) = -a^2 + 7 * a * b := by
  -- actual proof omitted
  sorry

example : (2 * (A 1 2) - (B 1 2)) = 13 := by
  -- actual proof omitted
  sorry

end simplify_expr_l843_84332


namespace faye_country_albums_l843_84357

theorem faye_country_albums (C : ℕ) (h1 : 6 * C + 18 = 30) : C = 2 :=
by
  -- This is the theorem statement with the necessary conditions and question
  sorry

end faye_country_albums_l843_84357


namespace second_quadrant_necessary_not_sufficient_l843_84303

variable (α : ℝ)

def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180
def is_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α < 180

theorem second_quadrant_necessary_not_sufficient : 
  (∀ α, is_obtuse α → is_second_quadrant α) ∧ ¬ (∀ α, is_second_quadrant α → is_obtuse α) := by
  sorry

end second_quadrant_necessary_not_sufficient_l843_84303


namespace total_overtime_hours_worked_l843_84377

def gary_wage : ℕ := 12
def mary_wage : ℕ := 14
def john_wage : ℕ := 16
def alice_wage : ℕ := 18
def michael_wage : ℕ := 20

def regular_hours : ℕ := 40
def overtime_rate : ℚ := 1.5

def total_paycheck : ℚ := 3646

theorem total_overtime_hours_worked :
  let gary_overtime := gary_wage * overtime_rate
  let mary_overtime := mary_wage * overtime_rate
  let john_overtime := john_wage * overtime_rate
  let alice_overtime := alice_wage * overtime_rate
  let michael_overtime := michael_wage * overtime_rate
  let regular_total := (gary_wage + mary_wage + john_wage + alice_wage + michael_wage) * regular_hours
  let total_overtime_pay := total_paycheck - regular_total
  let total_overtime_rate := gary_overtime + mary_overtime + john_overtime + alice_overtime + michael_overtime
  let overtime_hours := total_overtime_pay / total_overtime_rate
  overtime_hours.floor = 3 := 
by
  sorry

end total_overtime_hours_worked_l843_84377


namespace largest_n_multiple_of_7_l843_84342

theorem largest_n_multiple_of_7 (n : ℕ) (h1 : n < 50000) (h2 : (5*(n-3)^5 - 3*n^2 + 20*n - 35) % 7 = 0) : n = 49999 :=
sorry

end largest_n_multiple_of_7_l843_84342


namespace angle_of_inclination_l843_84363

theorem angle_of_inclination (t : ℝ) (x y : ℝ) :
  (x = 1 + t * (Real.sin (Real.pi / 6))) ∧ 
  (y = 2 + t * (Real.cos (Real.pi / 6))) →
  ∃ α : ℝ, α = Real.arctan (Real.sqrt 3) ∧ (0 ≤ α ∧ α < Real.pi) := 
by 
  sorry

end angle_of_inclination_l843_84363


namespace patriots_won_games_l843_84304

theorem patriots_won_games (C P M S T E : ℕ) 
  (hC : C > 25)
  (hPC : P > C)
  (hMP : M > P)
  (hSC : S > C)
  (hSP : S < P)
  (hTE : T > E) : 
  P = 35 :=
sorry

end patriots_won_games_l843_84304


namespace inequality_proof_l843_84385

theorem inequality_proof (a : ℝ) : (3 * a - 6) * (2 * a^2 - a^3) ≤ 0 := 
by 
  sorry

end inequality_proof_l843_84385


namespace f_three_equals_322_l843_84312

def f (z : ℝ) : ℝ := (z^2 - 2) * ((z^2 - 2)^2 - 3)

theorem f_three_equals_322 :
  f 3 = 322 :=
by
  -- Proof steps (left out intentionally as per instructions)
  sorry

end f_three_equals_322_l843_84312


namespace length_of_bridge_l843_84317

/-- Prove the length of the bridge -/
theorem length_of_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_sec : ℝ) : 
  train_length = 120 →
  train_speed_kmph = 70 →
  crossing_time_sec = 13.884603517432893 →
  (70 * (1000 / 3600) * 13.884603517432893 - 120 = 150) :=
by
  intros h1 h2 h3
  sorry

end length_of_bridge_l843_84317


namespace slower_pipe_time_l843_84321

/-
One pipe can fill a tank four times as fast as another pipe. 
If together the two pipes can fill the tank in 40 minutes, 
how long will it take for the slower pipe alone to fill the tank?
-/

theorem slower_pipe_time (t : ℕ) (h1 : ∀ t, 1/t + 4/t = 1/40) : t = 200 :=
sorry

end slower_pipe_time_l843_84321


namespace min_disks_needed_l843_84380

/-- 
  Sandhya must save 35 files onto disks, each with 1.44 MB space. 
  5 of the files take up 0.6 MB, 18 of the files take up 0.5 MB, 
  and the rest take up 0.3 MB. Files cannot be split across disks.
  Prove that the smallest number of disks needed to store all 35 files is 12.
--/
theorem min_disks_needed 
  (total_files : ℕ)
  (disk_capacity : ℝ)
  (file_sizes : ℕ → ℝ)
  (files_0_6_MB : ℕ)
  (files_0_5_MB : ℕ)
  (files_0_3_MB : ℕ)
  (remaining_files : ℕ)
  (storage_per_disk : ℝ)
  (smallest_disks_needed : ℕ) 
  (h1 : total_files = 35)
  (h2 : disk_capacity = 1.44)
  (h3 : file_sizes 0 = 0.6)
  (h4 : file_sizes 1 = 0.5)
  (h5 : file_sizes 2 = 0.3)
  (h6 : files_0_6_MB = 5)
  (h7 : files_0_5_MB = 18)
  (h8 : remaining_files = total_files - files_0_6_MB - files_0_5_MB)
  (h9 : remaining_files = 12)
  (h10 : storage_per_disk = file_sizes 0 * 2 + file_sizes 1 + file_sizes 2)
  (h11 : smallest_disks_needed = 12) :
  total_files = 35 ∧ disk_capacity = 1.44 ∧ storage_per_disk <= 1.44 ∧ smallest_disks_needed = 12 :=
by
  sorry

end min_disks_needed_l843_84380


namespace cost_per_mile_l843_84365

variable (x : ℝ)
variable (monday_miles : ℝ) (thursday_miles : ℝ) (base_cost : ℝ) (total_spent : ℝ)

-- Given conditions
def car_rental_conditions : Prop :=
  monday_miles = 620 ∧
  thursday_miles = 744 ∧
  base_cost = 150 ∧
  total_spent = 832 ∧
  total_spent = base_cost + (monday_miles + thursday_miles) * x

-- Theorem to prove the cost per mile
theorem cost_per_mile (h : car_rental_conditions x 620 744 150 832) : x = 0.50 :=
  by
    sorry

end cost_per_mile_l843_84365


namespace single_equivalent_discount_l843_84315

theorem single_equivalent_discount :
  let discount1 := 0.15
  let discount2 := 0.10
  let discount3 := 0.05
  ∃ (k : ℝ), (1 - k) = (1 - discount1) * (1 - discount2) * (1 - discount3) ∧ k = 0.27325 :=
by
  sorry

end single_equivalent_discount_l843_84315


namespace mean_score_seniors_138_l843_84371

def total_students : ℕ := 200
def mean_score_all : ℕ := 120

variable (s n : ℕ) -- number of seniors and non-seniors
variable (ms mn : ℚ) -- mean score of seniors and non-seniors

def non_seniors_twice_seniors := n = 2 * s
def mean_score_non_seniors := mn = 0.8 * ms
def total_students_eq := s + n = total_students

def total_score := (s : ℚ) * ms + (n : ℚ) * mn = (total_students : ℚ) * mean_score_all

theorem mean_score_seniors_138 :
  ∃ s n ms mn,
    non_seniors_twice_seniors s n ∧
    mean_score_non_seniors ms mn ∧
    total_students_eq s n ∧
    total_score s n ms mn → 
    ms = 138 :=
sorry

end mean_score_seniors_138_l843_84371


namespace tangent_intersect_x_axis_l843_84398

-- Defining the conditions based on the given problem
def radius1 : ℝ := 3
def center1 : ℝ × ℝ := (0, 0)

def radius2 : ℝ := 5
def center2 : ℝ × ℝ := (12, 0)

-- Stating what needs to be proved
theorem tangent_intersect_x_axis : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (∀ (x1 x2 : ℝ), 
    (x1 = x) ∧ 
    (x2 = 12 - x) ∧ 
    (radius1 / (center2.1 - x) = radius2 / x2) → 
    (x = 9 / 2)) := 
sorry

end tangent_intersect_x_axis_l843_84398


namespace solve_x_sq_plus_y_sq_l843_84323

theorem solve_x_sq_plus_y_sq (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = 2) : x^2 + y^2 = 5 :=
by
  sorry

end solve_x_sq_plus_y_sq_l843_84323


namespace average_of_rest_l843_84340

theorem average_of_rest (A : ℝ) (total_students scoring_95 scoring_0 : ℕ) (total_avg : ℝ)
  (h_total_students : total_students = 25)
  (h_scoring_95 : scoring_95 = 3)
  (h_scoring_0 : scoring_0 = 3)
  (h_total_avg : total_avg = 45.6)
  (h_sum_eq : total_students * total_avg = 3 * 95 + 3 * 0 + (total_students - scoring_95 - scoring_0) * A) :
  A = 45 := sorry

end average_of_rest_l843_84340


namespace average_weight_increase_l843_84368

variable (A N X : ℝ)

theorem average_weight_increase (hN : N = 135.5) (h_avg : A + X = (9 * A - 86 + N) / 9) : 
  X = 5.5 :=
by
  sorry

end average_weight_increase_l843_84368


namespace vasya_max_points_l843_84395

theorem vasya_max_points (cards : Finset (Fin 36)) 
  (petya_hand vasya_hand : Finset (Fin 36)) 
  (h_disjoint : Disjoint petya_hand vasya_hand)
  (h_union : petya_hand ∪ vasya_hand = cards)
  (h_card : cards.card = 36)
  (h_half : petya_hand.card = 18 ∧ vasya_hand.card = 18) : 
  ∃ max_points : ℕ, max_points = 15 := 
sorry

end vasya_max_points_l843_84395


namespace total_tv_show_cost_correct_l843_84359

noncomputable def total_cost_of_tv_show : ℕ :=
  let cost_per_episode_first_season := 100000
  let episodes_first_season := 12
  let episodes_seasons_2_to_4 := 18
  let cost_per_episode_other_seasons := 2 * cost_per_episode_first_season
  let episodes_last_season := 24
  let number_of_other_seasons := 4
  let total_cost_first_season := episodes_first_season * cost_per_episode_first_season
  let total_cost_other_seasons := (episodes_seasons_2_to_4 * 3 + episodes_last_season) * cost_per_episode_other_seasons
  total_cost_first_season + total_cost_other_seasons

theorem total_tv_show_cost_correct : total_cost_of_tv_show = 16800000 := by
  sorry

end total_tv_show_cost_correct_l843_84359


namespace total_number_of_glasses_l843_84331

open scoped Nat

theorem total_number_of_glasses (x y : ℕ) (h1 : y = x + 16) (h2 : (12 * x + 16 * y) / (x + y) = 15) : 12 * x + 16 * y = 480 := by
  sorry

end total_number_of_glasses_l843_84331


namespace solution_set_of_inequality_l843_84374

theorem solution_set_of_inequality : 
  { x : ℝ | x^2 - 3*x - 4 < 0 } = { x : ℝ | -1 < x ∧ x < 4 } :=
sorry

end solution_set_of_inequality_l843_84374


namespace integer_part_not_perfect_square_l843_84324

noncomputable def expr (n : ℕ) : ℝ :=
  2 * Real.sqrt (n + 1) / (Real.sqrt (n + 1) - Real.sqrt n)

theorem integer_part_not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = ⌊expr n⌋ :=
  sorry

end integer_part_not_perfect_square_l843_84324


namespace geometric_progression_product_l843_84390

variables {n : ℕ} {b q S S' P : ℝ} 

theorem geometric_progression_product (hb : b ≠ 0) (hq : q ≠ 1)
  (hP : P = b^n * q^(n*(n-1)/2))
  (hS : S = b * (1 - q^n) / (1 - q))
  (hS' : S' = (q^n - 1) / (b * (q - 1)))
  : P = (S * S')^(n/2) := 
sorry

end geometric_progression_product_l843_84390


namespace math_problem_l843_84392

theorem math_problem (d r : ℕ) (hd : d > 1)
  (h1 : 1259 % d = r) 
  (h2 : 1567 % d = r) 
  (h3 : 2257 % d = r) : d - r = 1 :=
by
  sorry

end math_problem_l843_84392


namespace Anne_wander_time_l843_84397

theorem Anne_wander_time (distance speed : ℝ) (h1 : distance = 3.0) (h2 : speed = 2.0) : distance / speed = 1.5 := by
  -- Given conditions
  sorry

end Anne_wander_time_l843_84397


namespace solve_x_l843_84330

def δ (x : ℝ) : ℝ := 4 * x + 6
def φ (x : ℝ) : ℝ := 5 * x + 4

theorem solve_x : ∃ x: ℝ, δ (φ x) = 3 → x = -19 / 20 := by
  sorry

end solve_x_l843_84330


namespace exists_func_satisfies_condition_l843_84388

theorem exists_func_satisfies_condition :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (x^2 + 2*x) = abs (x + 1) :=
sorry

end exists_func_satisfies_condition_l843_84388


namespace evaluate_expression_l843_84328

def diamond (a b : ℚ) : ℚ := a - (2 / b)

theorem evaluate_expression :
  ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -(11 / 30) :=
by
  sorry

end evaluate_expression_l843_84328


namespace findCorrectAnswer_l843_84345

-- Definitions
variable (x : ℕ)
def mistakenCalculation : Prop := 3 * x = 90
def correctAnswer : ℕ := x - 30

-- Theorem statement
theorem findCorrectAnswer (h : mistakenCalculation x) : correctAnswer x = 0 :=
sorry

end findCorrectAnswer_l843_84345


namespace rotated_translated_line_eq_l843_84320

theorem rotated_translated_line_eq :
  ∀ (x y : ℝ), y = 3 * x → y = - (1 / 3) * x + (1 / 3) :=
by
  sorry

end rotated_translated_line_eq_l843_84320


namespace oil_layer_height_l843_84302

/-- Given a tank with a rectangular bottom measuring 16 cm in length and 12 cm in width, initially containing 6 cm deep water and 6 cm deep oil, and an iron block with dimensions 8 cm in length, 8 cm in width, and 12 cm in height -/

theorem oil_layer_height (volume_water volume_oil volume_iron base_area new_volume_water : ℝ) 
  (base_area_def : base_area = 16 * 12) 
  (volume_water_def : volume_water = base_area * 6) 
  (volume_oil_def : volume_oil = base_area * 6) 
  (volume_iron_def : volume_iron = 8 * 8 * 12) 
  (new_volume_water_def : new_volume_water = volume_water + volume_iron) 
  (new_water_height : new_volume_water / base_area = 10) 
  : (volume_water + volume_oil) / base_area - (new_volume_water / base_area - 6) = 7 :=
by 
  sorry

end oil_layer_height_l843_84302


namespace algebraic_identity_l843_84343

theorem algebraic_identity (x y : ℝ) (h₁ : x * y = 4) (h₂ : x - y = 5) : 
  x^2 + 5 * x * y + y^2 = 53 := 
by 
  sorry

end algebraic_identity_l843_84343


namespace mrs_jane_total_coins_l843_84334

theorem mrs_jane_total_coins (Jayden_coins Jason_coins : ℕ) (h1 : Jayden_coins = 300) (h2 : Jason_coins = Jayden_coins + 60) :
  Jayden_coins + Jason_coins = 660 :=
sorry

end mrs_jane_total_coins_l843_84334


namespace work_earnings_t_l843_84352

theorem work_earnings_t (t : ℤ) (h1 : (t + 2) * (4 * t - 4) = (4 * t - 7) * (t + 3) + 3) : t = 10 := 
by
  sorry

end work_earnings_t_l843_84352


namespace factorization_correct_l843_84339

-- Define the given expression
def expression (a b : ℝ) : ℝ := 9 * a^2 * b - b

-- Define the factorized form
def factorized_form (a b : ℝ) : ℝ := b * (3 * a + 1) * (3 * a - 1)

-- Theorem stating that the factorization is correct
theorem factorization_correct (a b : ℝ) : expression a b = factorized_form a b := by
  sorry

end factorization_correct_l843_84339


namespace max_value_of_xy_l843_84300

theorem max_value_of_xy (x y : ℝ) (h₁ : x + y = 40) (h₂ : x > 0) (h₃ : y > 0) : xy ≤ 400 :=
sorry

end max_value_of_xy_l843_84300


namespace randy_quiz_goal_l843_84350

def randy_scores : List ℕ := [90, 98, 92, 94]
def randy_next_score : ℕ := 96
def randy_goal_average : ℕ := 94

theorem randy_quiz_goal :
  let total_score := randy_scores.sum
  let required_total_score := 470
  total_score + randy_next_score = required_total_score →
  required_total_score / randy_goal_average = 5 :=
by
  intro h
  sorry

end randy_quiz_goal_l843_84350


namespace joe_bought_books_l843_84393

theorem joe_bought_books (money_given : ℕ) (notebook_cost : ℕ) (num_notebooks : ℕ) (book_cost : ℕ) (leftover_money : ℕ) (total_spent := money_given - leftover_money) (spent_on_notebooks := num_notebooks * notebook_cost) (spent_on_books := total_spent - spent_on_notebooks) (num_books := spent_on_books / book_cost) : money_given = 56 → notebook_cost = 4 → num_notebooks = 7 → book_cost = 7 → leftover_money = 14 → num_books = 2 := by
  intros
  sorry

end joe_bought_books_l843_84393


namespace ramon_current_age_l843_84329

variable (R : ℕ) (L : ℕ)

theorem ramon_current_age :
  (L = 23) → (R + 20 = 2 * L) → R = 26 :=
by
  intro hL hR
  rw [hL] at hR
  have : R + 20 = 46 := by linarith
  linarith

end ramon_current_age_l843_84329


namespace ratio_of_speeds_l843_84319

-- Define the speeds V1 and V2
variable {V1 V2 : ℝ}

-- Given the initial conditions
def bike_ride_time_min := 10 -- in minutes
def subway_ride_time_min := 40 -- in minutes
def total_bike_only_time_min := 210 -- 3.5 hours in minutes

-- Prove the ratio of subway speed to bike speed is 5:1
theorem ratio_of_speeds (h : bike_ride_time_min * V1 + subway_ride_time_min * V2 = total_bike_only_time_min * V1) :
  V2 = 5 * V1 :=
by
  sorry

end ratio_of_speeds_l843_84319


namespace angle_of_inclination_l843_84310

theorem angle_of_inclination 
  (α : ℝ) 
  (h_tan : Real.tan α = -Real.sqrt 3)
  (h_range : 0 ≤ α ∧ α < 180) : α = 120 :=
by
  sorry

end angle_of_inclination_l843_84310


namespace avg_of_9_numbers_l843_84387

theorem avg_of_9_numbers (a b c d e f g h i : ℕ)
  (h1 : (a + b + c + d + e) / 5 = 99)
  (h2 : (e + f + g + h + i) / 5 = 100)
  (h3 : e = 59) : 
  (a + b + c + d + e + f + g + h + i) / 9 = 104 := 
sorry

end avg_of_9_numbers_l843_84387


namespace line_through_M_intersects_lines_l843_84355

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def line1 (t : ℝ) : Point3D :=
  {x := 2 - t, y := 3, z := -2 + t}

def plane1 (p : Point3D) : Prop :=
  2 * p.x - 2 * p.y - p.z - 4 = 0

def plane2 (p : Point3D) : Prop :=
  p.x + 3 * p.y + 2 * p.z + 1 = 0

def param_eq (t : ℝ) : Point3D :=
  {x := -2 + 13 * t, y := -3 * t, z := 3 - 12 * t}

theorem line_through_M_intersects_lines : 
  ∀ (t : ℝ), plane1 (param_eq t) ∧ plane2 (param_eq t) -> 
  ∃ t, param_eq t = {x := -2 + 13 * t, y := -3 * t, z := 3 - 12 * t} :=
by
  intros t h
  sorry

end line_through_M_intersects_lines_l843_84355


namespace jimmy_points_l843_84391

theorem jimmy_points (eng_pts init_eng_pts : ℕ) (math_pts init_math_pts : ℕ) 
  (sci_pts init_sci_pts : ℕ) (hist_pts init_hist_pts : ℕ) 
  (phy_pts init_phy_pts : ℕ) (eng_penalty math_penalty sci_penalty hist_penalty phy_penalty : ℕ)
  (passing_points : ℕ) (total_points_required : ℕ):
  init_eng_pts = 60 →
  init_math_pts = 55 →
  init_sci_pts = 40 →
  init_hist_pts = 70 →
  init_phy_pts = 50 →
  eng_penalty = 5 →
  math_penalty = 3 →
  sci_penalty = 8 →
  hist_penalty = 2 →
  phy_penalty = 6 →
  passing_points = 250 →
  total_points_required = (init_eng_pts - eng_penalty) + (init_math_pts - math_penalty) + 
                         (init_sci_pts - sci_penalty) + (init_hist_pts - hist_penalty) + 
                         (init_phy_pts - phy_penalty) →
  ∀ extra_loss, (total_points_required - extra_loss ≥ passing_points) → extra_loss ≤ 1 :=
by {
  sorry
}

end jimmy_points_l843_84391


namespace women_decreased_by_3_l843_84399

noncomputable def initial_men := 12
noncomputable def initial_women := 27

theorem women_decreased_by_3 
  (ratio_men_women : 4 / 5 = initial_men / initial_women)
  (men_after_enter : initial_men + 2 = 14)
  (women_after_leave : initial_women - 3 = 24) :
  (24 - 27 = -3) :=
by
  sorry

end women_decreased_by_3_l843_84399


namespace correct_calculation_l843_84325

theorem correct_calculation (a : ℝ) : -3 * a - 2 * a = -5 * a :=
by
  sorry

end correct_calculation_l843_84325


namespace eggs_not_eaten_is_6_l843_84381

noncomputable def eggs_not_eaten_each_week 
  (trays_purchased : ℕ) 
  (eggs_per_tray : ℕ) 
  (eggs_morning : ℕ) 
  (days_in_week : ℕ) 
  (eggs_night : ℕ) : ℕ :=
  let total_eggs := trays_purchased * eggs_per_tray
  let eggs_eaten_son_daughter := eggs_morning * days_in_week
  let eggs_eaten_rhea_husband := eggs_night * days_in_week
  let eggs_eaten_total := eggs_eaten_son_daughter + eggs_eaten_rhea_husband
  total_eggs - eggs_eaten_total

theorem eggs_not_eaten_is_6 
  (trays_purchased : ℕ := 2) 
  (eggs_per_tray : ℕ := 24) 
  (eggs_morning : ℕ := 2) 
  (days_in_week : ℕ := 7) 
  (eggs_night : ℕ := 4) : 
  eggs_not_eaten_each_week trays_purchased eggs_per_tray eggs_morning days_in_week eggs_night = 6 :=
by
  -- Here should be proof steps, but we use sorry to skip it as per instruction
  sorry

end eggs_not_eaten_is_6_l843_84381


namespace pure_gold_to_add_eq_46_67_l843_84383

-- Define the given conditions
variable (initial_alloy_weight : ℝ) (initial_gold_percentage : ℝ) (final_gold_percentage : ℝ)
variable (added_pure_gold : ℝ)

-- State the proof problem
theorem pure_gold_to_add_eq_46_67 :
  initial_alloy_weight = 20 ∧
  initial_gold_percentage = 0.50 ∧
  final_gold_percentage = 0.85 ∧
  (10 + added_pure_gold) / (20 + added_pure_gold) = 0.85 →
  added_pure_gold = 46.67 :=
by
  sorry

end pure_gold_to_add_eq_46_67_l843_84383


namespace fraction_mistake_l843_84347

theorem fraction_mistake (n : ℕ) (h : n = 288) (student_answer : ℕ) 
(h_student : student_answer = 240) : student_answer / n = 5 / 6 := 
by 
  -- Given that n = 288 and the student's answer = 240;
  -- we need to prove that 240/288 = 5/6
  sorry

end fraction_mistake_l843_84347


namespace compare_logs_l843_84349

noncomputable def a := Real.log 3
noncomputable def b := Real.log 3 / Real.log 2 / 2
noncomputable def c := Real.log 2 / Real.log 3 / 2

theorem compare_logs : a > b ∧ b > c := by
  sorry

end compare_logs_l843_84349


namespace total_cost_is_1_85_times_selling_price_l843_84314

def total_cost (P : ℝ) : ℝ := 140 * 2 * P + 90 * P

def loss (P : ℝ) : ℝ := 70 * 2 * P + 30 * P

def selling_price (P : ℝ) : ℝ := total_cost P - loss P

theorem total_cost_is_1_85_times_selling_price (P : ℝ) :
  total_cost P = 1.85 * selling_price P := by
  sorry

end total_cost_is_1_85_times_selling_price_l843_84314


namespace grasshopper_jumps_rational_angle_l843_84358

noncomputable def alpha_is_rational (α : ℝ) (jump : ℕ → ℝ × ℝ) : Prop :=
  ∃ k n : ℕ, (n ≠ 0) ∧ (jump n = (0, 0)) ∧ (α = (k : ℝ) / (n : ℝ) * 360)

theorem grasshopper_jumps_rational_angle :
  ∀ (α : ℝ) (jump : ℕ → ℝ × ℝ),
    (∀ n : ℕ, dist (jump (n + 1)) (jump n) = 1) →
    (jump 0 = (0, 0)) →
    (∃ n : ℕ, n ≠ 0 ∧ jump n = (0, 0)) →
    alpha_is_rational α jump :=
by
  intros α jump jumps_eq_1 start_exists returns_to_start
  sorry

end grasshopper_jumps_rational_angle_l843_84358


namespace evaluate_f_at_neg_three_l843_84333

def f (x : ℝ) : ℝ := 4 * x - 2

theorem evaluate_f_at_neg_three : f (-3) = -14 := by
  sorry

end evaluate_f_at_neg_three_l843_84333


namespace rebecca_groups_eq_l843_84348

-- Definitions
def total_eggs : ℕ := 15
def eggs_per_group : ℕ := 5
def expected_groups : ℕ := 3

-- Theorem to prove
theorem rebecca_groups_eq :
  total_eggs / eggs_per_group = expected_groups :=
by
  sorry

end rebecca_groups_eq_l843_84348


namespace aaronFoundCards_l843_84354

-- Given conditions
def initialCardsAaron : ℕ := 5
def finalCardsAaron : ℕ := 67

-- Theorem statement
theorem aaronFoundCards : finalCardsAaron - initialCardsAaron = 62 :=
by
  sorry

end aaronFoundCards_l843_84354


namespace polar_bear_daily_food_l843_84335

-- Definitions based on the conditions
def bucketOfTroutDaily : ℝ := 0.2
def bucketOfSalmonDaily : ℝ := 0.4

-- The proof statement
theorem polar_bear_daily_food : bucketOfTroutDaily + bucketOfSalmonDaily = 0.6 := by
  sorry

end polar_bear_daily_food_l843_84335


namespace geometric_series_sum_l843_84361

theorem geometric_series_sum :
  let a := (1 : ℚ) / 3
  let r := -(1 / 3)
  let n := 5
  let S₅ := (a * (1 - r ^ n)) / (1 - r)
  S₅ = 61 / 243 := by
  let a := (1 : ℚ) / 3
  let r := -(1 / 3)
  let n := 5
  let S₅ := (a * (1 - r ^ n)) / (1 - r)
  sorry

end geometric_series_sum_l843_84361


namespace prove_expression_l843_84316

theorem prove_expression (a : ℝ) (h : a^2 + a - 1 = 0) : 2 * a^2 + 2 * a + 2008 = 2010 := by
  sorry

end prove_expression_l843_84316


namespace pell_solution_unique_l843_84353

theorem pell_solution_unique 
  (x_0 y_0 x y : ℤ) 
  (h_fundamental : x_0^2 - 2003 * y_0^2 = 1)
  (h_pos_x : 0 < x) 
  (h_pos_y : 0 < y)
  (h_prime_div : ∀ p, Prime p → p ∣ x → p ∣ x_0) :
  x^2 - 2003 * y^2 = 1 → (x, y) = (x_0, y_0) :=
sorry

end pell_solution_unique_l843_84353


namespace find_a5_and_sum_l843_84386

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) > a n

-- Given conditions
def given_conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
is_geometric_sequence a q ∧ is_increasing_sequence a ∧ a 2 = 3 ∧ a 4 - a 3 = 18

-- Theorem to prove
theorem find_a5_and_sum {a : ℕ → ℝ} {q : ℝ} (h : given_conditions a q) :
  a 5 = 81 ∧ (a 1 + a 2 + a 3 + a 4 + a 5) = 121 :=
by
  -- Placeholder for the actual proof
  sorry

end find_a5_and_sum_l843_84386


namespace bailing_rate_bailing_problem_l843_84341

theorem bailing_rate (distance : ℝ) (rate_in : ℝ) (sink_limit : ℝ) (speed : ℝ) : ℝ :=
  let time_to_shore := distance / speed * 60 -- convert hours to minutes
  let total_intake := rate_in * time_to_shore
  let excess_water := total_intake - sink_limit
  excess_water / time_to_shore

theorem bailing_problem : bailing_rate 2 12 40 3 = 11 := by
  sorry

end bailing_rate_bailing_problem_l843_84341


namespace triangle_circle_fill_l843_84372

theorem triangle_circle_fill (A B C D : ℕ) : 
  (A ≠ B) → (A ≠ C) → (A ≠ D) → (B ≠ C) → (B ≠ D) → (C ≠ D) →
  (A = 6 ∨ A = 7 ∨ A = 8 ∨ A = 9) →
  (B = 6 ∨ B = 7 ∨ B = 8 ∨ B = 9) →
  (C = 6 ∨ C = 7 ∨ C = 8 ∨ C = 9) →
  (D = 6 ∨ D = 7 ∨ D = 8 ∨ D = 9) →
  (A + B + 1 + 8 =  A + 4 + 3 + 7) →  (D + 4 + 2 + 5 = 5 + 1 + 8 + B) →
  (5 + 1 + 8 + 6 = 5 + C + 7 + 4 ) →
  (A = 6) ∧ (B = 8) ∧ (C = 7) ∧ (D = 9) := by
  sorry

end triangle_circle_fill_l843_84372


namespace range_of_m_l843_84337

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (h_deriv : ∀ x, f' x < x)
variable (h_ineq : ∀ m, f (4 - m) - f m ≥ 8 - 4 * m)

theorem range_of_m (m : ℝ) : m ≥ 2 :=
sorry

end range_of_m_l843_84337


namespace sequence_contains_at_most_one_square_l843_84364

theorem sequence_contains_at_most_one_square 
  (a : ℕ → ℕ) 
  (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) : 
  ∀ m n, (m ≠ n) → ¬ (∃ k, a m = k^2 ∧ a n = k^2) :=
sorry

end sequence_contains_at_most_one_square_l843_84364


namespace no_prime_satisfies_condition_l843_84379

theorem no_prime_satisfies_condition :
  ¬ ∃ p : ℕ, p > 1 ∧ 10 * (p : ℝ) = (p : ℝ) + 5.4 := by {
  sorry
}

end no_prime_satisfies_condition_l843_84379


namespace probability_of_two_white_balls_correct_l843_84301

noncomputable def probability_of_two_white_balls : ℚ :=
  let total_balls := 15
  let white_balls := 8
  let first_draw_white := (white_balls : ℚ) / total_balls
  let second_draw_white := (white_balls - 1 : ℚ) / (total_balls - 1)
  first_draw_white * second_draw_white

theorem probability_of_two_white_balls_correct :
  probability_of_two_white_balls = 4 / 15 :=
by
  sorry

end probability_of_two_white_balls_correct_l843_84301


namespace dad_use_per_brush_correct_l843_84305

def toothpaste_total : ℕ := 105
def mom_use_per_brush : ℕ := 2
def anne_brother_use_per_brush : ℕ := 1
def brushing_per_day : ℕ := 3
def days_to_finish : ℕ := 5

-- Defining the daily use function for Anne's Dad
def dad_use_per_brush (D : ℕ) : ℕ := D

theorem dad_use_per_brush_correct (D : ℕ) 
  (h : brushing_per_day * (mom_use_per_brush + anne_brother_use_per_brush * 2 + dad_use_per_brush D) * days_to_finish = toothpaste_total) 
  : dad_use_per_brush D = 3 :=
by sorry

end dad_use_per_brush_correct_l843_84305


namespace problem_statement_l843_84394

theorem problem_statement (f : ℝ → ℝ) (a b : ℝ) (h₀ : ∀ x, f x = 4 * x + 3) (h₁ : a > 0) (h₂ : b > 0) :
  (∀ x, |f x + 5| < a ↔ |x + 3| < b) ↔ b ≤ a / 4 :=
sorry

end problem_statement_l843_84394


namespace membership_fee_increase_each_year_l843_84306

variable (fee_increase : ℕ)

def yearly_membership_fee_increase (first_year_fee sixth_year_fee yearly_increase : ℕ) : Prop :=
  yearly_increase * 5 = sixth_year_fee - first_year_fee

theorem membership_fee_increase_each_year :
  yearly_membership_fee_increase 80 130 10 :=
by
  unfold yearly_membership_fee_increase
  sorry

end membership_fee_increase_each_year_l843_84306


namespace circle_radius_of_tangent_parabolas_l843_84378

theorem circle_radius_of_tangent_parabolas :
  ∃ r : ℝ, 
  (∀ (x : ℝ), (x^2 + r = x)) →
  r = 1 / 4 :=
by
  sorry

end circle_radius_of_tangent_parabolas_l843_84378


namespace minimum_value_l843_84367

variable (m n x y : ℝ)

theorem minimum_value (h1 : m^2 + n^2 = 1) (h2 : x^2 + y^2 = 4) : 
  ∃ (min_val : ℝ), min_val = -2 ∧ ∀ (my_nx : ℝ), my_nx = my + nx → my_nx ≥ min_val :=
by
  sorry

end minimum_value_l843_84367


namespace find_number_l843_84375

theorem find_number (x : ℕ) (h : 15 * x = x + 196) : 15 * x = 210 :=
by
  sorry

end find_number_l843_84375


namespace slices_per_birthday_l843_84338

-- Define the conditions: 
-- k is the age, the number of candles, starting from 3.
variable (k : ℕ) (h : k ≥ 3)

-- Define the function for the number of triangular slices
def number_of_slices (k : ℕ) : ℕ := 2 * k - 5

-- State the theorem to prove that the number of slices is 2k - 5
theorem slices_per_birthday (k : ℕ) (h : k ≥ 3) : 
    number_of_slices k = 2 * k - 5 := 
by
  sorry

end slices_per_birthday_l843_84338


namespace unique_k_for_prime_roots_of_quadratic_l843_84376

/-- Function to check primality of a natural number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Theorem statement with the given conditions -/
theorem unique_k_for_prime_roots_of_quadratic :
  ∃! k : ℕ, ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p * q = k :=
sorry

end unique_k_for_prime_roots_of_quadratic_l843_84376


namespace find_t_l843_84346

-- Define the utility on both days
def utility_monday (t : ℝ) := t * (10 - t)
def utility_tuesday (t : ℝ) := (4 - t) * (t + 5)

-- Define the total hours spent on activities condition for both days
def total_hours_monday (t : ℝ) := t + (10 - t)
def total_hours_tuesday (t : ℝ) := (4 - t) + (t + 5)

theorem find_t : ∃ t : ℝ, t * (10 - t) = (4 - t) * (t + 5) ∧ 
                            total_hours_monday t ≥ 8 ∧ 
                            total_hours_tuesday t ≥ 8 :=
by
  sorry

end find_t_l843_84346


namespace slope_of_line_through_A_B_l843_84351

theorem slope_of_line_through_A_B :
  let A := (2, 1)
  let B := (-1, 3)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = -2/3 :=
by
  have A_x : Int := 2
  have A_y : Int := 1
  have B_x : Int := -1
  have B_y : Int := 3
  sorry

end slope_of_line_through_A_B_l843_84351


namespace original_cost_of_plants_l843_84313

theorem original_cost_of_plants
  (discount : ℕ)
  (amount_spent : ℕ)
  (original_cost : ℕ)
  (h_discount : discount = 399)
  (h_amount_spent : amount_spent = 68)
  (h_original_cost : original_cost = discount + amount_spent) :
  original_cost = 467 :=
by
  rw [h_discount, h_amount_spent] at h_original_cost
  exact h_original_cost

end original_cost_of_plants_l843_84313
