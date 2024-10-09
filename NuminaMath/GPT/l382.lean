import Mathlib

namespace silver_dollars_l382_38289

variable (C : ℕ)
variable (H : ℕ)
variable (P : ℕ)

theorem silver_dollars (h1 : H = P + 5) (h2 : P = C + 16) (h3 : C + P + H = 205) : C = 56 :=
by
  sorry

end silver_dollars_l382_38289


namespace domain_of_function_l382_38276

-- Define the setting and the constants involved
variables {f : ℝ → ℝ}
variable {c : ℝ}

-- The statement about the function's domain
theorem domain_of_function :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ (x ≤ 0 ∧ x ≠ -c) :=
sorry

end domain_of_function_l382_38276


namespace mechanic_hours_l382_38291

theorem mechanic_hours (h : ℕ) (labor_cost_per_hour parts_cost total_bill : ℕ) 
  (H1 : labor_cost_per_hour = 45) 
  (H2 : parts_cost = 225) 
  (H3 : total_bill = 450) 
  (H4 : labor_cost_per_hour * h + parts_cost = total_bill) : 
  h = 5 := 
by
  sorry

end mechanic_hours_l382_38291


namespace exponentiation_product_rule_l382_38237

theorem exponentiation_product_rule (a : ℝ) : (3 * a) ^ 2 = 9 * a ^ 2 :=
by
  sorry

end exponentiation_product_rule_l382_38237


namespace clearance_sale_gain_percent_l382_38208

theorem clearance_sale_gain_percent
  (SP : ℝ := 30)
  (gain_percent : ℝ := 25)
  (discount_percent : ℝ := 10)
  (CP : ℝ := SP/(1 + gain_percent/100)) :
  let Discount := discount_percent / 100 * SP
  let SP_sale := SP - Discount
  let Gain_during_sale := SP_sale - CP
  let Gain_percent_during_sale := (Gain_during_sale / CP) * 100
  Gain_percent_during_sale = 12.5 := 
by
  sorry

end clearance_sale_gain_percent_l382_38208


namespace best_model_based_on_R_squared_l382_38247

theorem best_model_based_on_R_squared:
  ∀ (R2_1 R2_2 R2_3 R2_4: ℝ), 
  R2_1 = 0.98 → R2_2 = 0.80 → R2_3 = 0.54 → R2_4 = 0.35 → 
  R2_1 ≥ R2_2 ∧ R2_1 ≥ R2_3 ∧ R2_1 ≥ R2_4 :=
by
  intros R2_1 R2_2 R2_3 R2_4 h1 h2 h3 h4
  sorry

end best_model_based_on_R_squared_l382_38247


namespace simplify_and_evaluate_equals_l382_38290

noncomputable def simplify_and_evaluate (a : ℝ) : ℝ :=
  (a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2) / (a^2 + 2 * a) / (a - 2)

theorem simplify_and_evaluate_equals (a : ℝ) (h : a^2 + 2 * a - 8 = 0) : 
  simplify_and_evaluate a = 1 / 4 :=
sorry

end simplify_and_evaluate_equals_l382_38290


namespace nth_equation_identity_l382_38286

theorem nth_equation_identity (n : ℕ) (h : n ≥ 1) : 
  (n / (n + 2 : ℚ)) * (1 - 1 / (n + 1 : ℚ)) = (n^2 / ((n + 1) * (n + 2) : ℚ)) := 
by 
  sorry

end nth_equation_identity_l382_38286


namespace solve_system_of_equations_l382_38269

theorem solve_system_of_equations (x y : ℚ) 
  (h1 : 3 * x - 2 * y = 8) 
  (h2 : x + 3 * y = 9) : 
  x = 42 / 11 ∧ y = 19 / 11 :=
by {
  sorry
}

end solve_system_of_equations_l382_38269


namespace irrational_neg_pi_lt_neg_two_l382_38216

theorem irrational_neg_pi_lt_neg_two (h1 : Irrational π) (h2 : π > 2) : Irrational (-π) ∧ -π < -2 := by
  sorry

end irrational_neg_pi_lt_neg_two_l382_38216


namespace baseball_cards_l382_38244

theorem baseball_cards (cards_per_page new_cards pages : ℕ) (h1 : cards_per_page = 8) (h2 : new_cards = 3) (h3 : pages = 2) : 
  (pages * cards_per_page - new_cards = 13) := by
  sorry

end baseball_cards_l382_38244


namespace trig_eqn_solution_l382_38205

open Real

theorem trig_eqn_solution (x : ℝ) (n : ℤ) :
  sin x ≠ 0 →
  cos x ≠ 0 →
  sin x + cos x ≥ 0 →
  (sqrt (1 + tan x) = sin x + cos x) →
  ∃ k : ℤ, (x = k * π + π / 4) ∨ (x = k * π - π / 4) ∨ (x = (2 * k * π + 3 * π / 4)) :=
by
  sorry

end trig_eqn_solution_l382_38205


namespace solve_g_eq_5_l382_38211

noncomputable def g (x : ℝ) : ℝ :=
if x < 0 then 4 * x + 8 else 3 * x - 15

theorem solve_g_eq_5 : {x : ℝ | g x = 5} = {-3/4, 20/3} :=
by
  sorry

end solve_g_eq_5_l382_38211


namespace line_exists_l382_38256

theorem line_exists (x y x' y' : ℝ)
  (h1 : x' = 3 * x + 2 * y + 1)
  (h2 : y' = x + 4 * y - 3) : 
  (∃ A B C : ℝ, A * x + B * y + C = 0 ∧ A * x' + B * y' + C = 0 ∧ 
  ((A = 1 ∧ B = -1 ∧ C = 4) ∨ (A = 4 ∧ B = -8 ∧ C = -5))) :=
sorry

end line_exists_l382_38256


namespace oranges_in_total_l382_38223

def number_of_boxes := 3
def oranges_per_box := 8
def total_oranges := 24

theorem oranges_in_total : number_of_boxes * oranges_per_box = total_oranges := 
by {
  -- sorry skips the proof part
  sorry 
}

end oranges_in_total_l382_38223


namespace find_certain_number_l382_38271

theorem find_certain_number (x : ℤ) (h : x - 5 = 4) : x = 9 :=
sorry

end find_certain_number_l382_38271


namespace solve_x_y_l382_38225

theorem solve_x_y (x y : ℝ) (h1 : x^2 + y^2 = 16 * x - 10 * y + 14) (h2 : x - y = 6) : 
  x + y = 3 := 
by 
  sorry

end solve_x_y_l382_38225


namespace volunteer_assigned_probability_l382_38253

theorem volunteer_assigned_probability :
  let volunteers := ["A", "B", "C", "D"]
  let areas := ["Beijing", "Zhangjiakou"]
  let total_ways := 14
  let favorable_ways := 6
  ∃ (p : ℚ), p = 6/14 → (1 / total_ways) * favorable_ways = 3/7
:= sorry

end volunteer_assigned_probability_l382_38253


namespace problem_1_problem_2_l382_38299

def M : Set ℕ := {0, 1}

def A := { p : ℕ × ℕ | p.fst ∈ M ∧ p.snd ∈ M }

def B := { p : ℕ × ℕ | p.snd = 1 - p.fst }

theorem problem_1 : A = {(0,0), (0,1), (1,0), (1,1)} :=
by
  sorry

theorem problem_2 : 
  let AB := { p ∈ A | p ∈ B }
  AB = {(1,0), (0,1)} ∧
  {S : Set (ℕ × ℕ) | S ⊆ AB} = {∅, {(1,0)}, {(0,1)}, {(1,0), (0,1)}} :=
by
  sorry

end problem_1_problem_2_l382_38299


namespace highest_possible_N_l382_38221

/--
In a football tournament with 15 teams, each team played exactly once against every other team.
A win earns 3 points, a draw earns 1 point, and a loss earns 0 points.
We need to prove that the highest possible integer \( N \) such that there are at least 6 teams with at least \( N \) points is 34.
-/
theorem highest_possible_N : 
  ∃ (N : ℤ) (teams : Fin 15 → ℤ) (successfulTeams : Fin 6 → Fin 15),
    (∀ i j, i ≠ j → teams i + teams j ≤ 207) ∧ 
    (∀ k, k < 6 → teams (successfulTeams k) ≥ 34) ∧ 
    (∀ k, 0 ≤ teams k) ∧ 
    N = 34 := sorry

end highest_possible_N_l382_38221


namespace quadratic_real_roots_l382_38275

theorem quadratic_real_roots (k : ℝ) (h1 : k ≠ 0) : (4 + 4 * k) ≥ 0 ↔ k ≥ -1 := 
by 
  sorry

end quadratic_real_roots_l382_38275


namespace not_all_pieces_found_l382_38258

theorem not_all_pieces_found (N : ℕ) (petya_tore : ℕ → ℕ) (vasya_tore : ℕ → ℕ) : 
  (∀ n, petya_tore n = n * 5 - n) →
  (∀ n, vasya_tore n = n * 9 - n) →
  1988 = N ∧ (N % 2 = 1) → false :=
by
  intros h_petya h_vasya h
  sorry

end not_all_pieces_found_l382_38258


namespace place_circle_no_overlap_l382_38229

theorem place_circle_no_overlap 
    (rect_width rect_height : ℝ) (num_squares : ℤ) (square_size square_diameter : ℝ)
    (h_rect_dims : rect_width = 20 ∧ rect_height = 25)
    (h_num_squares : num_squares = 120)
    (h_square_size : square_size = 1)
    (h_circle_diameter : square_diameter = 1) : 
  ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ rect_width ∧ 0 ≤ y ∧ y ≤ rect_height ∧ 
    ∀ (square_x square_y : ℝ), 
      0 ≤ square_x ∧ square_x ≤ rect_width - square_size ∧ 
      0 ≤ square_y ∧ square_y ≤ rect_height - square_size → 
      (x - square_x)^2 + (y - square_y)^2 ≥ (square_diameter / 2)^2 :=
sorry

end place_circle_no_overlap_l382_38229


namespace range_of_m_l382_38233

variable (x y m : ℝ)
variable (h1 : 0 < x)
variable (h2 : 0 < y)
variable (h3 : 2/x + 1/y = 1)
variable (h4 : ∀ x y : ℝ, x + 2*y > m^2 + 2*m)

theorem range_of_m (h1 : 0 < x) (h2 : 0 < y) (h3 : 2/x + 1/y = 1) (h4 : ∀ x y : ℝ, x + 2*y > m^2 + 2*m) : -4 < m ∧ m < 2 := 
sorry

end range_of_m_l382_38233


namespace find_particular_number_l382_38228

def particular_number (x : ℕ) : Prop :=
  (2 * (67 - (x / 23))) = 102

theorem find_particular_number : particular_number 2714 :=
by {
  sorry
}

end find_particular_number_l382_38228


namespace find_coefficient_m_l382_38297

theorem find_coefficient_m :
  ∃ m : ℝ, (1 + 2 * x)^3 = 1 + 6 * x + m * x^2 + 8 * x^3 ∧ m = 12 := by
  sorry

end find_coefficient_m_l382_38297


namespace speed_of_second_train_correct_l382_38263

noncomputable def length_first_train : ℝ := 140 -- in meters
noncomputable def length_second_train : ℝ := 160 -- in meters
noncomputable def time_to_cross : ℝ := 10.799136069114471 -- in seconds
noncomputable def speed_first_train : ℝ := 60 -- in km/hr
noncomputable def speed_second_train : ℝ := 40 -- in km/hr

theorem speed_of_second_train_correct :
  (length_first_train + length_second_train)/time_to_cross - (speed_first_train * (5/18)) = speed_second_train * (5/18) :=
by
  sorry

end speed_of_second_train_correct_l382_38263


namespace infinite_solutions_l382_38273

theorem infinite_solutions (a : ℤ) (h_a : a > 1) 
  (h_sol : ∃ x y : ℤ, x^2 - a * y^2 = -1) : 
  ∃ f : ℕ → ℤ × ℤ, ∀ n : ℕ, (f n).fst^2 - a * (f n).snd^2 = -1 :=
sorry

end infinite_solutions_l382_38273


namespace father_l382_38201

theorem father's_age : 
  ∀ (M F : ℕ), 
  (M = (2 : ℚ) / 5 * F) → 
  (M + 10 = (1 : ℚ) / 2 * (F + 10)) → 
  F = 50 :=
by
  intros M F h1 h2
  sorry

end father_l382_38201


namespace friends_attended_birthday_l382_38240

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l382_38240


namespace other_number_l382_38260

theorem other_number (x : ℕ) (h : 27 + x = 62) : x = 35 :=
by
  sorry

end other_number_l382_38260


namespace initial_gasoline_percentage_calculation_l382_38283

variable (initial_volume : ℝ)
variable (initial_ethanol_percentage : ℝ)
variable (additional_ethanol : ℝ)
variable (final_ethanol_percentage : ℝ)

theorem initial_gasoline_percentage_calculation
  (h1: initial_ethanol_percentage = 5)
  (h2: initial_volume = 45)
  (h3: additional_ethanol = 2.5)
  (h4: final_ethanol_percentage = 10) :
  100 - initial_ethanol_percentage = 95 :=
by
  sorry

end initial_gasoline_percentage_calculation_l382_38283


namespace probability_red_second_draw_l382_38272

theorem probability_red_second_draw 
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (after_first_draw_balls : ℕ)
  (after_first_draw_red : ℕ)
  (probability : ℚ) :
  total_balls = 5 →
  red_balls = 2 →
  white_balls = 3 →
  after_first_draw_balls = 4 →
  after_first_draw_red = 2 →
  probability = after_first_draw_red / after_first_draw_balls →
  probability = 0.5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end probability_red_second_draw_l382_38272


namespace total_modules_in_stock_l382_38203

-- Given conditions
def module_cost_high : ℝ := 10
def module_cost_low : ℝ := 3.5
def total_stock_value : ℝ := 45
def low_module_count : ℕ := 10

-- To be proved: total number of modules in stock
theorem total_modules_in_stock (x : ℕ) (y : ℕ) (h1 : y = low_module_count) 
  (h2 : module_cost_high * x + module_cost_low * y = total_stock_value) : 
  x + y = 11 := 
sorry

end total_modules_in_stock_l382_38203


namespace circle_diameter_l382_38248

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l382_38248


namespace set_of_points_l382_38236

theorem set_of_points (x y : ℝ) (h : x^2 * y - y ≥ 0) :
  (y ≥ 0 ∧ |x| ≥ 1) ∨ (y ≤ 0 ∧ |x| ≤ 1) :=
sorry

end set_of_points_l382_38236


namespace tent_ratio_l382_38282

-- Define the relevant variables
variables (N E S C T : ℕ)

-- State the conditions
def conditions : Prop :=
  N = 100 ∧
  E = 2 * N ∧
  S = 200 ∧
  T = 900 ∧
  N + E + S + C = T

-- State the theorem to prove the ratio
theorem tent_ratio (h : conditions N E S C T) : C = 4 * N :=
by sorry

end tent_ratio_l382_38282


namespace smallest_gcd_qr_l382_38268

theorem smallest_gcd_qr {p q r : ℕ} (hpq : Nat.gcd p q = 300) (hpr : Nat.gcd p r = 450) : 
  ∃ (g : ℕ), g = Nat.gcd q r ∧ g = 150 :=
by
  sorry

end smallest_gcd_qr_l382_38268


namespace explicit_formula_l382_38245

variable (f : ℝ → ℝ)
variable (is_quad : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
variable (max_value : ∀ x, f x ≤ 13)
variable (value_at_3 : f 3 = 5)
variable (value_at_neg1 : f (-1) = 5)

theorem explicit_formula :
  (∀ x, f x = -2 * x^2 + 4 * x + 11) :=
by
  sorry

end explicit_formula_l382_38245


namespace douglas_won_percentage_l382_38277

theorem douglas_won_percentage (p_X p_Y : ℝ) (r : ℝ) (V : ℝ) (h1 : p_X = 0.76) (h2 : p_Y = 0.4000000000000002) (h3 : r = 2) :
  (1.52 * V + 0.4000000000000002 * V) / (2 * V + V) * 100 = 64 := by
  sorry

end douglas_won_percentage_l382_38277


namespace new_member_money_l382_38212

variable (T M : ℝ)
variable (H1 : T / 7 = 20)
variable (H2 : (T + M) / 8 = 14)

theorem new_member_money : M = 756 :=
by
  sorry

end new_member_money_l382_38212


namespace division_of_neg6_by_3_l382_38224

theorem division_of_neg6_by_3 : (-6 : ℤ) / 3 = -2 := 
by
  sorry

end division_of_neg6_by_3_l382_38224


namespace positive_diff_40_x_l382_38250

theorem positive_diff_40_x
  (x : ℝ)
  (h : (40 + x + 15) / 3 = 35) :
  abs (x - 40) = 10 :=
sorry

end positive_diff_40_x_l382_38250


namespace f_max_min_l382_38218

def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom cauchy_f : ∀ x y : ℝ, f (x + y) = f x + f y
axiom less_than_zero : ∀ x : ℝ, x > 0 → f x < 0
axiom f_one : f 1 = -2

theorem f_max_min : (∀ x ∈ [-3, 3], f (-3) = 6 ∧ f 3 = -6) :=
by sorry

end f_max_min_l382_38218


namespace onion_rings_cost_l382_38270

variable (hamburger_cost smoothie_cost total_payment change_received : ℕ)

theorem onion_rings_cost (h_hamburger : hamburger_cost = 4) 
                         (h_smoothie : smoothie_cost = 3) 
                         (h_total_payment : total_payment = 20) 
                         (h_change_received : change_received = 11) :
                         total_payment - change_received - hamburger_cost - smoothie_cost = 2 :=
by
  sorry

end onion_rings_cost_l382_38270


namespace seconds_in_12_5_minutes_l382_38264

theorem seconds_in_12_5_minutes :
  let minutes := 12.5
  let seconds_per_minute := 60
  minutes * seconds_per_minute = 750 :=
by
  let minutes := 12.5
  let seconds_per_minute := 60
  sorry

end seconds_in_12_5_minutes_l382_38264


namespace egg_production_difference_l382_38252

-- Define the conditions
def last_year_production : ℕ := 1416
def this_year_production : ℕ := 4636

-- Define the theorem statement
theorem egg_production_difference :
  this_year_production - last_year_production = 3220 :=
by
  sorry

end egg_production_difference_l382_38252


namespace max_students_equal_distribution_l382_38210

-- Define the number of pens and pencils
def pens : ℕ := 1008
def pencils : ℕ := 928

-- Define the problem statement which asks for the GCD of the given numbers
theorem max_students_equal_distribution : Nat.gcd pens pencils = 16 :=
by 
  -- Lean's gcd computation can be used to confirm the result
  sorry

end max_students_equal_distribution_l382_38210


namespace maximum_value_of_f_l382_38217

def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem maximum_value_of_f :
  ∀ (a : ℝ), (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x a ≥ -2) → f 2 a = 25 :=
by
  intro a h
  -- sorry to skip the proof
  sorry

end maximum_value_of_f_l382_38217


namespace two_digit_number_eq_27_l382_38296

theorem two_digit_number_eq_27 (A : ℕ) (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
    (h : A = 10 * x + y) (hcond : A = 3 * (x + y)) : A = 27 :=
by
  sorry

end two_digit_number_eq_27_l382_38296


namespace candles_left_in_room_l382_38261

-- Define the variables and conditions
def total_candles : ℕ := 40
def alyssa_used : ℕ := total_candles / 2
def remaining_candles_after_alyssa : ℕ := total_candles - alyssa_used
def chelsea_used : ℕ := (7 * remaining_candles_after_alyssa) / 10
def final_remaining_candles : ℕ := remaining_candles_after_alyssa - chelsea_used

-- The theorem we need to prove
theorem candles_left_in_room : final_remaining_candles = 6 := by
  sorry

end candles_left_in_room_l382_38261


namespace shape_described_by_theta_eq_c_is_plane_l382_38215

-- Definitions based on conditions in the problem
def spherical_coordinates (ρ θ φ : ℝ) := true

def is_plane_condition (θ c : ℝ) := θ = c

-- Statement to prove
theorem shape_described_by_theta_eq_c_is_plane (c : ℝ) :
  ∀ ρ θ φ : ℝ, spherical_coordinates ρ θ φ → is_plane_condition θ c → "Plane" = "Plane" :=
by sorry

end shape_described_by_theta_eq_c_is_plane_l382_38215


namespace A_inter_B_eq_A_A_union_B_l382_38292

-- Definitions for sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + 3 * a = (a + 3) * x}
def B : Set ℝ := {x | x^2 + 3 = 4 * x}

-- Proof problem for part (1)
theorem A_inter_B_eq_A (a : ℝ) : (A a ∩ B = A a) ↔ (a = 1 ∨ a = 3) :=
by
  sorry

-- Proof problem for part (2)
theorem A_union_B (a : ℝ) : A a ∪ B = if a = 1 then {1, 3} else if a = 3 then {1, 3} else {a, 1, 3} :=
by
  sorry

end A_inter_B_eq_A_A_union_B_l382_38292


namespace greatest_divisor_l382_38213

theorem greatest_divisor :
  ∃ x, (∀ y : ℕ, y > 0 → x ∣ (7^y + 12*y - 1)) ∧ (∀ z, (∀ y : ℕ, y > 0 → z ∣ (7^y + 12*y - 1)) → z ≤ x) ∧ x = 18 :=
sorry

end greatest_divisor_l382_38213


namespace number_of_pencils_selling_price_equals_loss_l382_38284

theorem number_of_pencils_selling_price_equals_loss :
  ∀ (S C L : ℝ) (N : ℕ),
  C = 1.3333333333333333 * S →
  L = C - S →
  (S / 60) * N = L →
  N = 20 :=
by
  intros S C L N hC hL hN
  sorry

end number_of_pencils_selling_price_equals_loss_l382_38284


namespace sum_of_areas_of_circles_l382_38280

theorem sum_of_areas_of_circles :
  (∑' n : ℕ, π * (9 / 16) ^ n) = π * (16 / 7) :=
by
  sorry

end sum_of_areas_of_circles_l382_38280


namespace pairs_satisfy_equation_l382_38214

theorem pairs_satisfy_equation :
  ∀ (x n : ℕ), (x > 0 ∧ n > 0) ∧ 3 * 2 ^ x + 4 = n ^ 2 → (x, n) = (2, 4) ∨ (x, n) = (5, 10) ∨ (x, n) = (6, 14) :=
by
  sorry

end pairs_satisfy_equation_l382_38214


namespace trigonometric_expression_value_l382_38294

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 5 := by
  sorry

end trigonometric_expression_value_l382_38294


namespace eiffel_tower_vs_burj_khalifa_l382_38266

-- Define the heights of the structures
def height_eiffel_tower : ℕ := 324
def height_burj_khalifa : ℕ := 830

-- Define the statement to be proven
theorem eiffel_tower_vs_burj_khalifa :
  height_burj_khalifa - height_eiffel_tower = 506 :=
by
  sorry

end eiffel_tower_vs_burj_khalifa_l382_38266


namespace Nancy_folders_l382_38234

def n_initial : ℕ := 43
def n_deleted : ℕ := 31
def n_per_folder : ℕ := 6
def n_folders : ℕ := (n_initial - n_deleted) / n_per_folder

theorem Nancy_folders : n_folders = 2 := by
  sorry

end Nancy_folders_l382_38234


namespace estimate_total_fish_in_pond_l382_38222

theorem estimate_total_fish_in_pond :
  ∀ (total_tagged_fish initial_sample_size second_sample_size tagged_in_second_sample : ℕ),
  initial_sample_size = 100 →
  second_sample_size = 200 →
  tagged_in_second_sample = 10 →
  total_tagged_fish = 100 →
  (total_tagged_fish : ℚ) / (total_fish : ℚ) = tagged_in_second_sample / second_sample_size →
  total_fish = 2000 := by
  intros total_tagged_fish initial_sample_size second_sample_size tagged_in_second_sample
  intro h1 h2 h3 h4 h5
  sorry

end estimate_total_fish_in_pond_l382_38222


namespace number_of_correct_answers_l382_38220

def total_questions := 30
def correct_points := 3
def incorrect_points := -1
def total_score := 78

theorem number_of_correct_answers (x : ℕ) :
  3 * x + incorrect_points * (total_questions - x) = total_score → x = 27 :=
by
  sorry

end number_of_correct_answers_l382_38220


namespace clay_weight_in_second_box_l382_38295

/-- Define the properties of the first and second boxes -/
structure Box where
  height : ℕ
  width : ℕ
  length : ℕ
  weight : ℕ

noncomputable def box1 : Box :=
  { height := 2, width := 3, length := 5, weight := 40 }

noncomputable def box2 : Box :=
  { height := 2 * 2, width := 3 * 3, length := 5, weight := 240 }

theorem clay_weight_in_second_box : 
  box2.weight = (box2.height * box2.width * box2.length) / 
                (box1.height * box1.width * box1.length) * box1.weight :=
by
  sorry

end clay_weight_in_second_box_l382_38295


namespace systematic_sampling_third_group_number_l382_38288

theorem systematic_sampling_third_group_number :
  ∀ (total_members groups sample_number group_5_number group_gap : ℕ),
  total_members = 200 →
  groups = 40 →
  sample_number = total_members / groups →
  group_5_number = 22 →
  group_gap = 5 →
  (group_this_number : ℕ) = group_5_number - (5 - 3) * group_gap →
  group_this_number = 12 :=
by
  intros total_members groups sample_number group_5_number group_gap Htotal Hgroups Hsample Hgroup5 Hgap Hthis_group
  sorry

end systematic_sampling_third_group_number_l382_38288


namespace bicycle_final_price_l382_38209

theorem bicycle_final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (h1 : original_price = 200) (h2 : discount1 = 0.4) (h3 : discount2 = 0.2) :
  (original_price * (1 - discount1) * (1 - discount2)) = 96 :=
by
  -- sorry proof here
  sorry

end bicycle_final_price_l382_38209


namespace functional_eq_solution_l382_38243

noncomputable def f : ℝ → ℝ := sorry

theorem functional_eq_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) →
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_eq_solution_l382_38243


namespace day_90_N_minus_1_is_Thursday_l382_38227

/-- 
    Given that the 150th day of year N is a Sunday, 
    and the 220th day of year N+2 is also a Sunday,
    prove that the 90th day of year N-1 is a Thursday.
-/
theorem day_90_N_minus_1_is_Thursday (N : ℕ)
    (h1 : (150 % 7 = 0))  -- 150th day of year N is Sunday
    (h2 : (220 % 7 = 0))  -- 220th day of year N + 2 is Sunday
    : ((90 + 366) % 7 = 4) := -- 366 days in a leap year (N-1), 90th day modulo 7 is Thursday
by
  sorry

end day_90_N_minus_1_is_Thursday_l382_38227


namespace trajectory_of_P_l382_38219

def point := ℝ × ℝ

-- Definitions for points A and F, and the circle equation
def A : point := (-1, 0)
def F (x y : ℝ) := (x - 1) ^ 2 + y ^ 2 = 16

-- Main theorem statement: proving the trajectory equation of point P
theorem trajectory_of_P : 
  (∀ (B : point), F B.1 B.2 → 
  (∃ P : point, ∃ (k : ℝ), (P.1 - B.1) * k = -(P.2 - B.2) ∧ (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0)) →
  (∃ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1) :=
sorry

end trajectory_of_P_l382_38219


namespace smallest_number_condition_l382_38206

theorem smallest_number_condition 
  (x : ℕ) 
  (h1 : ∃ k : ℕ, x - 6 = k * 12)
  (h2 : ∃ k : ℕ, x - 6 = k * 16)
  (h3 : ∃ k : ℕ, x - 6 = k * 18)
  (h4 : ∃ k : ℕ, x - 6 = k * 21)
  (h5 : ∃ k : ℕ, x - 6 = k * 28)
  (h6 : ∃ k : ℕ, x - 6 = k * 35)
  (h7 : ∃ k : ℕ, x - 6 = k * 39) 
  : x = 65526 :=
sorry

end smallest_number_condition_l382_38206


namespace composite_integer_divisors_l382_38231

theorem composite_integer_divisors (n : ℕ) (k : ℕ) (d : ℕ → ℕ) 
  (h_composite : 1 < n ∧ ¬Prime n)
  (h_divisors : ∀ i, 1 ≤ i ∧ i ≤ k → d i ∣ n)
  (h_distinct : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → d i < d j)
  (h_range : d 1 = 1 ∧ d k = n)
  (h_ratio : ∀ i, 1 ≤ i ∧ i < k → (d (i + 1) - d i) = (i * (d 2 - d 1))) : n = 6 :=
by sorry

end composite_integer_divisors_l382_38231


namespace x_equals_neg_x_is_zero_abs_x_equals_2_is_pm_2_l382_38287

theorem x_equals_neg_x_is_zero (x : ℝ) (h : x = -x) : x = 0 := sorry

theorem abs_x_equals_2_is_pm_2 (x : ℝ) (h : |x| = 2) : x = 2 ∨ x = -2 := sorry

end x_equals_neg_x_is_zero_abs_x_equals_2_is_pm_2_l382_38287


namespace euclidean_division_mod_l382_38285

theorem euclidean_division_mod (h1 : 2022 % 19 = 8)
                               (h2 : 8^6 % 19 = 1)
                               (h3 : 2023 % 6 = 1)
                               (h4 : 2023^2024 % 6 = 1) 
: 2022^(2023^2024) % 19 = 8 := 
by
  sorry

end euclidean_division_mod_l382_38285


namespace andrew_paid_total_l382_38207

-- Define the quantities and rates
def quantity_grapes : ℕ := 14
def rate_grapes : ℕ := 54
def quantity_mangoes : ℕ := 10
def rate_mangoes : ℕ := 62

-- Define the cost calculations
def cost_grapes : ℕ := quantity_grapes * rate_grapes
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes
def total_cost : ℕ := cost_grapes + cost_mangoes

-- Prove the total amount paid is as expected
theorem andrew_paid_total : total_cost = 1376 := by
  sorry 

end andrew_paid_total_l382_38207


namespace find_a_2016_l382_38204

-- Given definition for the sequence sum
def sequence_sum (n : ℕ) : ℕ := n * n

-- Definition for a_n using the given sequence sum
def term (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

-- Stating the theorem that we need to prove
theorem find_a_2016 : term 2016 = 4031 := 
by 
  sorry

end find_a_2016_l382_38204


namespace arithmetic_sequence_n_equals_100_l382_38246

theorem arithmetic_sequence_n_equals_100
  (a₁ : ℕ) (d : ℕ) (a_n : ℕ)
  (h₁ : a₁ = 1)
  (h₂ : d = 3)
  (h₃ : a_n = 298) :
  ∃ n : ℕ, a_n = a₁ + (n - 1) * d ∧ n = 100 :=
by
  sorry

end arithmetic_sequence_n_equals_100_l382_38246


namespace sum_of_angles_of_circumscribed_quadrilateral_l382_38255

theorem sum_of_angles_of_circumscribed_quadrilateral
  (EF GH : ℝ)
  (EF_central_angle : EF = 100)
  (GH_central_angle : GH = 120) :
  (EF / 2 + GH / 2) = 70 :=
by
  sorry

end sum_of_angles_of_circumscribed_quadrilateral_l382_38255


namespace find_k_l382_38259

theorem find_k (x : ℝ) (k : ℝ) (h : 2 * x - 3 = 3 * x - 2 + k) (h_solution : x = 2) : k = -3 := by
  sorry

end find_k_l382_38259


namespace g_g_g_3_eq_71_l382_38230

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 2 * n - 1 else 2 * n + 5

theorem g_g_g_3_eq_71 : g (g (g 3)) = 71 := 
by
  sorry

end g_g_g_3_eq_71_l382_38230


namespace lunch_break_is_48_minutes_l382_38265

noncomputable def lunch_break_duration (L : ℝ) (p a : ℝ) : Prop :=
  (8 - L) * (p + a) = 0.6 ∧ 
  (9 - L) * p = 0.35 ∧
  (5 - L) * a = 0.1

theorem lunch_break_is_48_minutes :
  ∃ L p a, lunch_break_duration L p a ∧ L * 60 = 48 :=
by
  -- proof steps would go here
  sorry

end lunch_break_is_48_minutes_l382_38265


namespace tailwind_speed_l382_38249

-- Define the given conditions
def plane_speed_with_wind (P W : ℝ) : Prop := P + W = 460
def plane_speed_against_wind (P W : ℝ) : Prop := P - W = 310

-- Theorem stating the proof problem
theorem tailwind_speed (P W : ℝ) 
  (h1 : plane_speed_with_wind P W) 
  (h2 : plane_speed_against_wind P W) : 
  W = 75 :=
sorry

end tailwind_speed_l382_38249


namespace find_a10_l382_38279

theorem find_a10 (a : ℕ → ℚ) 
  (h1 : a 1 = 1/2)
  (h2 : ∀ n : ℕ, 0 < n → (1 / (a (n + 1) - 1)) = (1 / (a n - 1)) - 1) : 
  a 10 = 10/11 := 
sorry

end find_a10_l382_38279


namespace range_of_m_l382_38274

-- Defining the conditions
variable (x m : ℝ)

-- The theorem statement
theorem range_of_m (h : ∀ x : ℝ, x < m → 2*x + 1 < 5) : m ≤ 2 := by
  sorry

end range_of_m_l382_38274


namespace joe_flight_expense_l382_38254

theorem joe_flight_expense
  (initial_amount : ℕ)
  (hotel_expense : ℕ)
  (food_expense : ℕ)
  (remaining_amount : ℕ)
  (flight_expense : ℕ)
  (h1 : initial_amount = 6000)
  (h2 : hotel_expense = 800)
  (h3 : food_expense = 3000)
  (h4 : remaining_amount = 1000)
  (h5 : flight_expense = initial_amount - remaining_amount - hotel_expense - food_expense) :
  flight_expense = 1200 :=
by
  sorry

end joe_flight_expense_l382_38254


namespace solution_set_bf_x2_solution_set_g_l382_38262

def f (x : ℝ) := x^2 - 5 * x + 6

theorem solution_set_bf_x2 (x : ℝ) : (2 < x ∧ x < 3) ↔ f x < 0 := sorry

noncomputable def g (x : ℝ) := 6 * x^2 - 5 * x + 1

theorem solution_set_g (x : ℝ) : (1 / 3 < x ∧ x < 1 / 2) ↔ g x < 0 := sorry

end solution_set_bf_x2_solution_set_g_l382_38262


namespace number_of_candies_l382_38202

theorem number_of_candies (n : ℕ) (h1 : 11 ≤ n) (h2 : n ≤ 100) (h3 : n % 18 = 0) (h4 : n % 7 = 1) : n = 36 :=
by
  sorry

end number_of_candies_l382_38202


namespace second_percentage_increase_l382_38257

theorem second_percentage_increase :
  ∀ (P : ℝ) (x : ℝ), (P * 1.30 * (1 + x) = P * 1.5600000000000001) → x = 0.2 :=
by
  intros P x h
  sorry

end second_percentage_increase_l382_38257


namespace no_real_solution_abs_eq_quadratic_l382_38298

theorem no_real_solution_abs_eq_quadratic (x : ℝ) : abs (2 * x - 6) ≠ x^2 - x + 2 := by
  sorry

end no_real_solution_abs_eq_quadratic_l382_38298


namespace find_ellipse_parameters_l382_38238

noncomputable def ellipse_centers_and_axes (F1 F2 : ℝ × ℝ) (d : ℝ) (tangent_slope : ℝ) :=
  let h := (F1.1 + F2.1) / 2
  let k := (F1.2 + F2.2) / 2
  let a := d / 2
  let c := (Real.sqrt ((F2.1 - F1.1)^2 + (F2.2 - F1.2)^2)) / 2
  let b := Real.sqrt (a^2 - c^2)
  (h, k, a, b)

theorem find_ellipse_parameters :
  let F1 := (-1, 1)
  let F2 := (5, 1)
  let d := 10
  let tangent_at_x_axis_slope := 1
  let (h, k, a, b) := ellipse_centers_and_axes F1 F2 d tangent_at_x_axis_slope
  h + k + a + b = 12 :=
by
  sorry

end find_ellipse_parameters_l382_38238


namespace multiplicative_inverse_of_550_mod_4319_l382_38267

theorem multiplicative_inverse_of_550_mod_4319 :
  (48^2 + 275^2 = 277^2) → ((550 * 2208) % 4319 = 1) := by
  intro h
  sorry

end multiplicative_inverse_of_550_mod_4319_l382_38267


namespace slope_of_line_AB_l382_38241

-- Define the points A and B
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (2, 4)

-- State the proposition that we need to prove
theorem slope_of_line_AB :
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = 5 / 2 := by
  sorry

end slope_of_line_AB_l382_38241


namespace sophist_statements_correct_l382_38200

-- Definitions based on conditions
def num_knights : ℕ := 40
def num_liars : ℕ := 25

-- Statements made by the sophist
def sophist_statement1 : Prop := ∃ (sophist : Prop), sophist ∧ sophist → num_knights = 40
def sophist_statement2 : Prop := ∃ (sophist : Prop), sophist ∧ sophist → num_liars + 1 = 26

-- Theorem to be proved
theorem sophist_statements_correct :
  sophist_statement1 ∧ sophist_statement2 :=
by
  -- Placeholder for the actual proof
  sorry

end sophist_statements_correct_l382_38200


namespace weight_of_dried_grapes_l382_38235

/-- The weight of dried grapes available from 20 kg of fresh grapes given the water content in fresh and dried grapes. -/
theorem weight_of_dried_grapes (W_fresh W_dried : ℝ) (fresh_weight : ℝ) (weight_dried : ℝ) :
  W_fresh = 0.9 → 
  W_dried = 0.2 → 
  fresh_weight = 20 →
  weight_dried = (0.1 * fresh_weight) / (1 - W_dried) → 
  weight_dried = 2.5 :=
by sorry

end weight_of_dried_grapes_l382_38235


namespace hexagon_diagonals_l382_38242

theorem hexagon_diagonals (n : ℕ) (h : n = 6) : (n * (n - 3)) / 2 = 9 := by
  sorry

end hexagon_diagonals_l382_38242


namespace student_B_speed_l382_38232

theorem student_B_speed 
  (distance : ℝ)
  (time_difference : ℝ)
  (speed_ratio : ℝ)
  (B_speed A_speed : ℝ) 
  (h_distance : distance = 12)
  (h_time_difference : time_difference = 10 / 60) -- 10 minutes in hours
  (h_speed_ratio : A_speed = 1.2 * B_speed)
  (h_A_time : distance / A_speed = distance / B_speed - time_difference)
  : B_speed = 12 := sorry

end student_B_speed_l382_38232


namespace John_overall_profit_l382_38239

theorem John_overall_profit :
  let CP_grinder := 15000
  let Loss_percentage_grinder := 0.04
  let CP_mobile_phone := 8000
  let Profit_percentage_mobile_phone := 0.10
  let CP_refrigerator := 24000
  let Profit_percentage_refrigerator := 0.08
  let CP_television := 12000
  let Loss_percentage_television := 0.06
  let SP_grinder := CP_grinder * (1 - Loss_percentage_grinder)
  let SP_mobile_phone := CP_mobile_phone * (1 + Profit_percentage_mobile_phone)
  let SP_refrigerator := CP_refrigerator * (1 + Profit_percentage_refrigerator)
  let SP_television := CP_television * (1 - Loss_percentage_television)
  let Total_CP := CP_grinder + CP_mobile_phone + CP_refrigerator + CP_television
  let Total_SP := SP_grinder + SP_mobile_phone + SP_refrigerator + SP_television
  let Overall_profit := Total_SP - Total_CP
  Overall_profit = 1400 := by
  sorry

end John_overall_profit_l382_38239


namespace sum_of_youngest_and_oldest_friend_l382_38293

-- Given definitions
def mean_age_5 := 12
def median_age_5 := 11
def one_friend_age := 10

-- The total sum of ages is given by mean * number of friends
def total_sum_ages : ℕ := 5 * mean_age_5

-- Third friend's age as defined by median
def third_friend_age := 11

-- Proving the sum of the youngest and oldest friend's ages
theorem sum_of_youngest_and_oldest_friend:
  (∃ youngest oldest : ℕ, youngest + oldest = 38) :=
by
  sorry

end sum_of_youngest_and_oldest_friend_l382_38293


namespace sum_youngest_oldest_l382_38278

-- Define the ages of the cousins
variables (a1 a2 a3 a4 : ℕ)

-- Conditions given in the problem
def mean_age (a1 a2 a3 a4 : ℕ) : Prop := (a1 + a2 + a3 + a4) / 4 = 8
def median_age (a2 a3 : ℕ) : Prop := (a2 + a3) / 2 = 5

-- Main theorem statement to be proved
theorem sum_youngest_oldest (h_mean : mean_age a1 a2 a3 a4) (h_median : median_age a2 a3) :
  a1 + a4 = 22 :=
sorry

end sum_youngest_oldest_l382_38278


namespace unique_solution_abs_eq_l382_38251

theorem unique_solution_abs_eq : ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| :=
by
  sorry

end unique_solution_abs_eq_l382_38251


namespace circle_equation_l382_38226

-- Definitions for the given conditions
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (-1, 1)
def line (p : ℝ × ℝ) : Prop := p.1 + p.2 - 2 = 0

-- Theorem statement for the proof problem
theorem circle_equation :
  ∃ (h k : ℝ), line (h, k) ∧ (h = 1) ∧ (k = 1) ∧
  ((h - 1)^2 + (k - 1)^2 = 4) :=
sorry

end circle_equation_l382_38226


namespace annual_average_growth_rate_estimated_output_value_2006_l382_38281

-- First problem: Prove the annual average growth rate from 2003 to 2005
theorem annual_average_growth_rate (x : ℝ) (h : 6.4 * (1 + x)^2 = 10) : 
  x = 1/4 :=
by
  sorry

-- Second problem: Prove the estimated output value for 2006 given the annual growth rate
theorem estimated_output_value_2006 (x : ℝ) (output_2005 : ℝ) (h_growth : x = 1/4) (h_2005 : output_2005 = 10) : 
  output_2005 * (1 + x) = 12.5 :=
by 
  sorry

end annual_average_growth_rate_estimated_output_value_2006_l382_38281
