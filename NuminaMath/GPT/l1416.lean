import Mathlib

namespace value_of_expression_l1416_141641

theorem value_of_expression (n : ℕ) (a : ℝ) (h1 : 6 * 11 * n ≠ 0) (h2 : a ^ (2 * n) = 5) : 2 * a ^ (6 * n) - 4 = 246 :=
by
  sorry

end value_of_expression_l1416_141641


namespace pieces_bound_l1416_141691

open Finset

variable {n : ℕ} (B W : ℕ)

theorem pieces_bound (n : ℕ) (B W : ℕ) (hB : B ≤ n^2) (hW : W ≤ n^2) :
    B ≤ n^2 ∨ W ≤ n^2 := 
by
  sorry

end pieces_bound_l1416_141691


namespace sum_first_7_terms_is_105_l1416_141678

-- Define an arithmetic sequence
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Given conditions
variables {a d : ℕ}
axiom a4_is_15 : arithmetic_seq a d 4 = 15

-- Goal/theorem to be proven
theorem sum_first_7_terms_is_105 : sum_arithmetic_seq a d 7 = 105 :=
sorry

end sum_first_7_terms_is_105_l1416_141678


namespace balloons_total_l1416_141682

theorem balloons_total (number_of_groups balloons_per_group : ℕ)
  (h1 : number_of_groups = 7) (h2 : balloons_per_group = 5) : 
  number_of_groups * balloons_per_group = 35 := by
  sorry

end balloons_total_l1416_141682


namespace bandage_overlap_l1416_141636

theorem bandage_overlap
  (n : ℕ) (l : ℝ) (total_length : ℝ) (required_length : ℝ)
  (h_n : n = 20) (h_l : l = 15.25) (h_required_length : required_length = 248) :
  (required_length = l * n - (n - 1) * 3) :=
by
  sorry

end bandage_overlap_l1416_141636


namespace min_value_of_x_under_conditions_l1416_141676

noncomputable def S (x y z : ℝ) : ℝ := (z + 1)^2 / (2 * x * y * z)

theorem min_value_of_x_under_conditions :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x^2 + y^2 + z^2 = 1 →
  (∃ x_min : ℝ, S x y z = S x_min x_min (Real.sqrt 2 - 1) ∧ x_min = Real.sqrt (Real.sqrt 2 - 1)) :=
by
  intros x y z hx hy hz hxyz
  use Real.sqrt (Real.sqrt 2 - 1)
  sorry

end min_value_of_x_under_conditions_l1416_141676


namespace sum_of_squares_eq_power_l1416_141698

theorem sum_of_squares_eq_power (n : ℕ) : ∃ x y z : ℕ, x^2 + y^2 = z^n :=
sorry

end sum_of_squares_eq_power_l1416_141698


namespace tank_holds_gallons_l1416_141642

noncomputable def tank_initial_fraction := (7 : ℚ) / 8
noncomputable def tank_partial_fraction := (2 : ℚ) / 3
def gallons_used := 15

theorem tank_holds_gallons
  (x : ℚ) -- number of gallons the tank holds when full
  (h_initial : tank_initial_fraction * x - gallons_used = tank_partial_fraction * x) :
  x = 72 := 
sorry

end tank_holds_gallons_l1416_141642


namespace intersection_P_Q_l1416_141692

def P (x : ℝ) : Prop := x + 2 ≥ x^2

def Q (x : ℕ) : Prop := x ≤ 3

theorem intersection_P_Q :
  {x : ℕ | P x} ∩ {x : ℕ | Q x} = {0, 1, 2} :=
by
  sorry

end intersection_P_Q_l1416_141692


namespace neg_pi_lt_neg_three_l1416_141626

theorem neg_pi_lt_neg_three (h : Real.pi > 3) : -Real.pi < -3 :=
sorry

end neg_pi_lt_neg_three_l1416_141626


namespace max_marks_eq_300_l1416_141612

-- Problem Statement in Lean 4

theorem max_marks_eq_300 (m_score p_score c_score : ℝ) 
    (m_percent p_percent c_percent : ℝ)
    (h1 : m_score = 285) (h2 : m_percent = 95) 
    (h3 : p_score = 270) (h4 : p_percent = 90) 
    (h5 : c_score = 255) (h6 : c_percent = 85) :
    (m_score / (m_percent / 100) = 300) ∧ 
    (p_score / (p_percent / 100) = 300) ∧ 
    (c_score / (c_percent / 100) = 300) :=
by
  sorry

end max_marks_eq_300_l1416_141612


namespace ben_current_age_l1416_141647

theorem ben_current_age (a b c : ℕ) 
  (h1 : a + b + c = 36) 
  (h2 : c = 2 * a - 4) 
  (h3 : b + 5 = 3 * (a + 5) / 4) : 
  b = 5 := 
by
  sorry

end ben_current_age_l1416_141647


namespace total_matches_played_l1416_141689

def home_team_wins := 3
def home_team_draws := 4
def home_team_losses := 0
def rival_team_wins := 2 * home_team_wins
def rival_team_draws := home_team_draws
def rival_team_losses := 0

theorem total_matches_played :
  home_team_wins + home_team_draws + home_team_losses + rival_team_wins + rival_team_draws + rival_team_losses = 17 :=
by
  sorry

end total_matches_played_l1416_141689


namespace fourth_vertex_exists_l1416_141675

structure Point :=
  (x : ℚ)
  (y : ℚ)

def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_parallelogram (A B C D : Point) : Prop :=
  let M_AC := Point.mk ((A.x + C.x) / 2) ((A.y + C.y) / 2)
  let M_BD := Point.mk ((B.x + D.x) / 2) ((B.y + D.y) / 2)
  is_midpoint M_AC A C ∧ is_midpoint M_BD B D ∧ M_AC = M_BD

theorem fourth_vertex_exists (A B C : Point) (hA : A = ⟨-1, 0⟩) (hB : B = ⟨3, 0⟩) (hC : C = ⟨1, -5⟩) :
  ∃ D : Point, (D = ⟨1, 5⟩ ∨ D = ⟨-3, -5⟩) ∧ is_parallelogram A B C D :=
by
  sorry

end fourth_vertex_exists_l1416_141675


namespace Mark_sold_1_box_less_than_n_l1416_141688

variable (M A n : ℕ)

theorem Mark_sold_1_box_less_than_n (h1 : n = 8)
 (h2 : A = n - 2)
 (h3 : M + A < n)
 (h4 : M ≥ 1) 
 (h5 : A ≥ 1)
 : M = 1 := 
sorry

end Mark_sold_1_box_less_than_n_l1416_141688


namespace sin_theta_val_sin_2theta_pi_div_6_val_l1416_141640

open Real

theorem sin_theta_val (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2) 
  (hcos : cos (θ + π / 6) = 1 / 3) : 
  sin θ = (2 * sqrt 6 - 1) / 6 := 
by sorry

theorem sin_2theta_pi_div_6_val (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (hcos : cos (θ + π / 6) = 1 / 3) : 
  sin (2 * θ + π / 6) = (4 * sqrt 6 + 7) / 18 := 
by sorry

end sin_theta_val_sin_2theta_pi_div_6_val_l1416_141640


namespace real_and_imag_parts_of_z_l1416_141673

noncomputable def real_part (z : ℂ) : ℝ := z.re
noncomputable def imag_part (z : ℂ) : ℝ := z.im

theorem real_and_imag_parts_of_z :
  ∀ (i : ℂ), i * i = -1 → 
  ∀ (z : ℂ), z = i * (-1 + 2 * i) → real_part z = -2 ∧ imag_part z = -1 :=
by 
  intros i hi z hz
  sorry

end real_and_imag_parts_of_z_l1416_141673


namespace hyperbola_foci_eccentricity_l1416_141611

-- Definitions and conditions
def hyperbola_eq := (x y : ℝ) → (x^2 / 4) - (y^2 / 12) = 1

-- Proof goals: Coordinates of the foci and eccentricity
theorem hyperbola_foci_eccentricity (x y : ℝ) : 
  (∃ c : ℝ, (x^2 / 4) - (y^2 / 12) = 1 ∧ (x = 4 ∧ y = 0) ∨ (x = -4 ∧ y = 0)) ∧ 
  (∃ e : ℝ, e = 2) :=
sorry

end hyperbola_foci_eccentricity_l1416_141611


namespace Tim_has_16_pencils_l1416_141666

variable (T_Sarah T_Tyrah T_Tim : Nat)

-- Conditions
def condition1 : Prop := T_Tyrah = 6 * T_Sarah
def condition2 : Prop := T_Tim = 8 * T_Sarah
def condition3 : Prop := T_Tyrah = 12

-- Theorem to prove
theorem Tim_has_16_pencils (h1 : condition1 T_Sarah T_Tyrah) (h2 : condition2 T_Sarah T_Tim) (h3 : condition3 T_Tyrah) : T_Tim = 16 :=
by
  sorry

end Tim_has_16_pencils_l1416_141666


namespace breadth_of_rectangular_plot_l1416_141667

theorem breadth_of_rectangular_plot (b l : ℝ) (h1 : l = 3 * b) (h2 : l * b = 432) : b = 12 := 
sorry

end breadth_of_rectangular_plot_l1416_141667


namespace trumpet_cost_l1416_141609

def cost_of_song_book : Real := 5.84
def total_spent : Real := 151
def cost_of_trumpet : Real := total_spent - cost_of_song_book

theorem trumpet_cost : cost_of_trumpet = 145.16 :=
by
  sorry

end trumpet_cost_l1416_141609


namespace probability_two_black_balls_l1416_141607

theorem probability_two_black_balls (white_balls black_balls drawn_balls : ℕ) 
  (h_w : white_balls = 4) (h_b : black_balls = 7) (h_d : drawn_balls = 2) :
  let total_ways := Nat.choose (white_balls + black_balls) drawn_balls
  let black_ways := Nat.choose black_balls drawn_balls
  (black_ways / total_ways : ℚ) = 21 / 55 :=
by
  sorry

end probability_two_black_balls_l1416_141607


namespace exp_f_f_increasing_inequality_l1416_141671

noncomputable def f (a b : ℝ) (x : ℝ) :=
  (a * x + b) / (x^2 + 1)

-- Conditions
variable (a b : ℝ)
axiom h_odd : ∀ x : ℝ, f a b (-x) = - f a b x
axiom h_value : f a b (1/2) = 2/5

-- Proof statements
theorem exp_f : f a b x = x / (x^2 + 1) := sorry

theorem f_increasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : 
  f a b x1 < f a b x2 := sorry

theorem inequality (x : ℝ) (h1 : 0 < x) (h2 : x < 1/3) :
  f a b (2 * x - 1) + f a b x < 0 := sorry

end exp_f_f_increasing_inequality_l1416_141671


namespace find_pumpkin_seed_packets_l1416_141660

variable (P : ℕ)

-- Problem assumptions (conditions)
def pumpkin_seed_cost : ℝ := 2.50
def tomato_seed_cost_total : ℝ := 1.50 * 4
def chili_pepper_seed_cost_total : ℝ := 0.90 * 5
def total_spent : ℝ := 18.00

-- Main theorem to prove
theorem find_pumpkin_seed_packets (P : ℕ) (h : (pumpkin_seed_cost * P) + tomato_seed_cost_total + chili_pepper_seed_cost_total = total_spent) : P = 3 := by sorry

end find_pumpkin_seed_packets_l1416_141660


namespace find_central_cell_l1416_141699

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end find_central_cell_l1416_141699


namespace least_five_digit_congruent_to_six_mod_seventeen_l1416_141627

theorem least_five_digit_congruent_to_six_mod_seventeen : ∃ x : ℕ, x ≥ 10000 ∧ x < 100000 ∧ x % 17 = 6 ∧ ∀ y : ℕ, y ≥ 10000 ∧ y < 100000 ∧ y % 17 = 6 → x ≤ y :=
  by
    sorry

end least_five_digit_congruent_to_six_mod_seventeen_l1416_141627


namespace pentagonal_tiles_count_l1416_141693

theorem pentagonal_tiles_count (t p : ℕ) (h1 : t + p = 30) (h2 : 3 * t + 5 * p = 100) : p = 5 :=
sorry

end pentagonal_tiles_count_l1416_141693


namespace correct_statements_about_opposite_numbers_l1416_141652

/-- Definition of opposite numbers: two numbers are opposite if one is the negative of the other --/
def is_opposite (a b : ℝ) : Prop := a = -b

theorem correct_statements_about_opposite_numbers (a b : ℝ) :
  (is_opposite a b ↔ a + b = 0) ∧
  (a + b = 0 ↔ is_opposite a b) ∧
  ((is_opposite a b ∧ a ≠ 0 ∧ b ≠ 0) ↔ (a / b = -1)) ∧
  ((a / b = -1 ∧ b ≠ 0) ↔ is_opposite a b) :=
by {
  sorry -- Proof is omitted
}

end correct_statements_about_opposite_numbers_l1416_141652


namespace missing_dog_number_l1416_141677

theorem missing_dog_number {S : Finset ℕ} (h₁ : S =  Finset.range 25 \ {24}) (h₂ : S.sum id = 276) :
  (∃ y ∈ S, y = (S.sum id - y) / (S.card - 1)) ↔ 24 ∉ S :=
by
  sorry

end missing_dog_number_l1416_141677


namespace shorter_leg_right_triangle_l1416_141625

theorem shorter_leg_right_triangle (a b c : ℕ) (h0 : a^2 + b^2 = c^2) (h1 : c = 39) (h2 : a < b) : a = 15 :=
by {
  sorry
}

end shorter_leg_right_triangle_l1416_141625


namespace convert_C_to_F_l1416_141608

theorem convert_C_to_F (C F : ℝ) (h1 : C = 40) (h2 : C = 5 / 9 * (F - 32)) : F = 104 := 
by
  -- Proof goes here
  sorry

end convert_C_to_F_l1416_141608


namespace solve_system_of_odes_l1416_141655

theorem solve_system_of_odes (C₁ C₂ : ℝ) :
  ∃ (x y : ℝ → ℝ),
    (∀ t, x t = (C₁ + C₂ * t) * Real.exp (3 * t)) ∧
    (∀ t, y t = (C₁ + C₂ + C₂ * t) * Real.exp (3 * t)) ∧
    (∀ t, deriv x t = 2 * x t + y t) ∧
    (∀ t, deriv y t = 4 * y t - x t) :=
by
  sorry

end solve_system_of_odes_l1416_141655


namespace incorrect_option_C_l1416_141616

theorem incorrect_option_C (a b d : ℝ) (h₁ : ∀ x : ℝ, x ≠ d → x^2 + a * x + b > 0) (h₂ : a > 0) :
  ¬∀ x₁ x₂ : ℝ, (x₁ * x₂ > 0) → ((x₁, x₂) ∈ {p : (ℝ × ℝ) | p.1^2 + a * p.1 - b < 0 ∧ p.2^2 + a * p.2 - b < 0}) :=
sorry

end incorrect_option_C_l1416_141616


namespace pages_with_money_l1416_141680

def cost_per_page : ℝ := 3.5
def total_money : ℝ := 15 * 100

theorem pages_with_money : ⌊total_money / cost_per_page⌋ = 428 :=
by sorry

end pages_with_money_l1416_141680


namespace remainder_of_first_105_sum_div_5280_l1416_141659

theorem remainder_of_first_105_sum_div_5280:
  let n := 105
  let d := 5280
  let sum := n * (n + 1) / 2
  sum % d = 285 := by
  sorry

end remainder_of_first_105_sum_div_5280_l1416_141659


namespace evaluate_g_at_3_l1416_141643

def g (x : ℝ) := 3 * x ^ 4 - 5 * x ^ 3 + 4 * x ^ 2 - 7 * x + 2

theorem evaluate_g_at_3 : g 3 = 125 :=
by
  -- Proof omitted for this exercise.
  sorry

end evaluate_g_at_3_l1416_141643


namespace option_A_correct_l1416_141658

theorem option_A_correct (x y : ℝ) (hy : y ≠ 0) :
  (-2 * x^2 * y + y) / y = -2 * x^2 + 1 :=
by
  sorry

end option_A_correct_l1416_141658


namespace Jim_remaining_distance_l1416_141628

theorem Jim_remaining_distance (t d r : ℕ) (h₁ : t = 1200) (h₂ : d = 923) (h₃ : r = t - d) : r = 277 := 
by 
  -- Proof steps would go here
  sorry

end Jim_remaining_distance_l1416_141628


namespace jellybeans_red_l1416_141623

-- Define the individual quantities of each color of jellybean.
def b := 14
def p := 26
def o := 40
def pk := 7
def y := 21
def T := 237

-- Prove that the number of red jellybeans is 129.
theorem jellybeans_red : T - (b + p + o + pk + y) = 129 := by
  -- (optional: you can include intermediate steps if needed, but it's not required here)
  sorry

end jellybeans_red_l1416_141623


namespace betty_gave_stuart_percentage_l1416_141687

theorem betty_gave_stuart_percentage (P : ℝ) 
  (betty_marbles : ℝ := 60) 
  (stuart_initial_marbles : ℝ := 56) 
  (stuart_final_marbles : ℝ := 80)
  (increase_in_stuart_marbles : ℝ := stuart_final_marbles - stuart_initial_marbles)
  (betty_to_stuart : ℝ := (P / 100) * betty_marbles) :
  56 + ((P / 100) * betty_marbles) = 80 → P = 40 :=
by
  intros h
  -- Sorry is used since the proof steps are not required
  sorry

end betty_gave_stuart_percentage_l1416_141687


namespace problem_solution_l1416_141686

def count_multiples_of_5_not_15 : ℕ := 
  let count_up_to (m n : ℕ) := n / m
  let multiples_of_5_up_to_300 := count_up_to 5 299
  let multiples_of_15_up_to_300 := count_up_to 15 299
  multiples_of_5_up_to_300 - multiples_of_15_up_to_300

theorem problem_solution : count_multiples_of_5_not_15 = 40 := by
  sorry

end problem_solution_l1416_141686


namespace gcd_of_four_sum_1105_l1416_141606

theorem gcd_of_four_sum_1105 (a b c d : ℕ) (h_sum : a + b + c + d = 1105)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (hd_pos : 0 < d)
  (h_neq_ab : a ≠ b) (h_neq_ac : a ≠ c) (h_neq_ad : a ≠ d)
  (h_neq_bc : b ≠ c) (h_neq_bd : b ≠ d) (h_neq_cd : c ≠ d)
  (h_gcd_ab : gcd a b > 1) (h_gcd_ac : gcd a c > 1) (h_gcd_ad : gcd a d > 1)
  (h_gcd_bc : gcd b c > 1) (h_gcd_bd : gcd b d > 1) (h_gcd_cd : gcd c d > 1) :
  gcd a (gcd b (gcd c d)) = 221 := by
  sorry

end gcd_of_four_sum_1105_l1416_141606


namespace water_glass_ounces_l1416_141670

theorem water_glass_ounces (glasses_per_day : ℕ) (days_per_week : ℕ)
    (bottle_ounces : ℕ) (bottle_fills_per_week : ℕ)
    (total_glasses_per_week : ℕ)
    (total_ounces_per_week : ℕ)
    (glasses_per_week_eq : glasses_per_day * days_per_week = total_glasses_per_week)
    (ounces_per_week_eq : bottle_ounces * bottle_fills_per_week = total_ounces_per_week)
    (ounce_per_glass : ℕ)
    (glasses_per_week : ℕ)
    (ounces_per_week : ℕ) :
    total_ounces_per_week / total_glasses_per_week = 5 :=
by
  sorry

end water_glass_ounces_l1416_141670


namespace new_population_difference_l1416_141684

def population_eagles : ℕ := 150
def population_falcons : ℕ := 200
def population_hawks : ℕ := 320
def population_owls : ℕ := 270
def increase_rate : ℕ := 10

theorem new_population_difference :
  let least_populous := min population_eagles (min population_falcons (min population_hawks population_owls))
  let most_populous := max population_eagles (max population_falcons (max population_hawks population_owls))
  let increased_least_populous := least_populous + least_populous * increase_rate / 100
  most_populous - increased_least_populous = 155 :=
by
  sorry

end new_population_difference_l1416_141684


namespace part1_part2_l1416_141681

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x * abs (x^2 - a)

-- Define the two main proofs to be shown
theorem part1 (a : ℝ) (h : a = 1) : 
  ∃ I1 I2 : Set ℝ, I1 = Set.Icc (-1 - Real.sqrt 2) (-1) ∧ I2 = Set.Icc (-1 + Real.sqrt 2) (1) ∧ 
  ∀ x ∈ I1 ∪ I2, ∀ y ∈ I1 ∪ I2, x ≤ y → f y 1 ≤ f x 1 :=
sorry

theorem part2 (a : ℝ) (h : a ≥ 0) (h_roots : ∀ m : ℝ, (∃ x : ℝ, x > 0 ∧ f x a = m) ∧ (∃ x : ℝ, x < 0 ∧ f x a = m)) : 
  ∃ m : ℝ, m = 4 / (Real.exp 2) :=
sorry

end part1_part2_l1416_141681


namespace solve_for_a_l1416_141644

noncomputable def special_otimes (a b : ℝ) : ℝ :=
  if a > b then a^2 + b else a + b^2

theorem solve_for_a (a : ℝ) : special_otimes a (-2) = 4 → a = Real.sqrt 6 :=
by
  intro h
  sorry

end solve_for_a_l1416_141644


namespace sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine_l1416_141668

theorem sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine : 
  Real.sqrt (3^3 + 3^3 + 3^3) = 9 :=
by 
  sorry

end sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine_l1416_141668


namespace num_birds_is_six_l1416_141600

-- Define the number of nests
def N : ℕ := 3

-- Define the difference between the number of birds and nests
def diff : ℕ := 3

-- Prove that the number of birds is 6
theorem num_birds_is_six (B : ℕ) (h1 : N = 3) (h2 : B - N = diff) : B = 6 := by
  -- Placeholder for the proof
  sorry

end num_birds_is_six_l1416_141600


namespace log_exp_identity_l1416_141601

noncomputable def a : ℝ := Real.log 3 / Real.log 4

theorem log_exp_identity : 2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 := 
by
  sorry

end log_exp_identity_l1416_141601


namespace fraction_equality_l1416_141661

variable (a b : ℚ)

theorem fraction_equality (h : (4 * a + 3 * b) / (4 * a - 3 * b) = 4) : a / b = 5 / 4 := by
  sorry

end fraction_equality_l1416_141661


namespace consecutive_odd_numbers_l1416_141619

/- 
  Out of some consecutive odd numbers, 9 times the first number 
  is equal to the addition of twice the third number and adding 9 
  to twice the second. Let x be the first number, then we aim to prove that 
  9 * x = 2 * (x + 4) + 2 * (x + 2) + 9 ⟹ x = 21 / 5
-/

theorem consecutive_odd_numbers (x : ℚ) (h : 9 * x = 2 * (x + 4) + 2 * (x + 2) + 9) : x = 21 / 5 :=
sorry

end consecutive_odd_numbers_l1416_141619


namespace distinct_balls_boxes_l1416_141634

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l1416_141634


namespace paths_H_to_J_via_I_l1416_141662

def binom (n k : ℕ) : ℕ := Nat.choose n k

def paths_from_H_to_I : ℕ :=
  binom 7 2  -- Calculate the number of paths from H(0,7) to I(5,5)

def paths_from_I_to_J : ℕ :=
  binom 8 3  -- Calculate the number of paths from I(5,5) to J(8,0)

theorem paths_H_to_J_via_I : paths_from_H_to_I * paths_from_I_to_J = 1176 := by
  -- This theorem states that the number of paths from H to J through I is 1176
  sorry  -- Proof to be provided

end paths_H_to_J_via_I_l1416_141662


namespace compute_product_l1416_141651

theorem compute_product (x1 y1 x2 y2 x3 y3 : ℝ) 
  (h1 : x1^3 - 3 * x1 * y1^2 = 1005) 
  (h2 : y1^3 - 3 * x1^2 * y1 = 1004)
  (h3 : x2^3 - 3 * x2 * y2^2 = 1005)
  (h4 : y2^3 - 3 * x2^2 * y2 = 1004)
  (h5 : x3^3 - 3 * x3 * y3^2 = 1005)
  (h6 : y3^3 - 3 * x3^2 * y3 = 1004) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 502 := 
sorry

end compute_product_l1416_141651


namespace find_a_l1416_141630

theorem find_a (a : ℝ) (h : 6 * a + 4 = 0) : a = -2 / 3 :=
by
  sorry

end find_a_l1416_141630


namespace find_line_equation_l1416_141620

-- Definitions: Point and Line in 2D
structure Point2D where
  x : ℝ
  y : ℝ

structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Line passes through the point
def line_through_point (L : Line2D) (P : Point2D) : Prop :=
  L.a * P.x + L.b * P.y + L.c = 0

-- Perpendicular lines condition: if Line L1 and Line L2 are perpendicular.
def perpendicular (L1 L2 : Line2D) : Prop :=
  L1.a * L2.a + L1.b * L2.b = 0

-- Define line1 and line2 as given
def line1 : Line2D := {a := 1, b := -2, c := 0} -- corresponds to x - 2y + m = 0

-- Define point P (-1, 3)
def P : Point2D := {x := -1, y := 3}

-- Required line passing through point P and perpendicular to line1
def required_line : Line2D := {a := 2, b := 1, c := -1}

-- The proof goal
theorem find_line_equation : (line_through_point required_line P) ∧ (perpendicular line1 required_line) :=
by
  sorry

end find_line_equation_l1416_141620


namespace remainder_when_divided_by_x_minus_3_l1416_141633

def p (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7

theorem remainder_when_divided_by_x_minus_3 : p 3 = 52 := 
by
  -- proof here
  sorry

end remainder_when_divided_by_x_minus_3_l1416_141633


namespace perfect_square_condition_l1416_141646

theorem perfect_square_condition (x m : ℝ) (h : ∃ k : ℝ, x^2 + x + 2*m = k^2) : m = 1/8 := 
sorry

end perfect_square_condition_l1416_141646


namespace max_gcd_b_n_b_n_plus_1_max_possible_value_of_e_n_l1416_141697

def b_n (n : ℕ) : ℤ := (10 ^ n - 9) / 3
def e_n (n : ℕ) : ℤ := Int.gcd (b_n n) (b_n (n + 1))

theorem max_gcd_b_n_b_n_plus_1 : ∀ n : ℕ, e_n n ≤ 3 :=
by
  -- Provide the proof here
  sorry

theorem max_possible_value_of_e_n : ∃ n : ℕ, e_n n = 3 :=
by
  -- Provide the proof here
  sorry

end max_gcd_b_n_b_n_plus_1_max_possible_value_of_e_n_l1416_141697


namespace string_length_is_correct_l1416_141656

noncomputable def calculate_string_length (circumference height : ℝ) (loops : ℕ) : ℝ :=
  let vertical_distance_per_loop := height / loops
  let hypotenuse_length := Real.sqrt ((circumference ^ 2) + (vertical_distance_per_loop ^ 2))
  loops * hypotenuse_length

theorem string_length_is_correct : calculate_string_length 6 16 5 = 34 := 
  sorry

end string_length_is_correct_l1416_141656


namespace value_of_4b_minus_a_l1416_141632

theorem value_of_4b_minus_a (a b : ℕ) (h1 : a > b) (h2 : x^2 - 20*x + 96 = (x - a)*(x - b)) : 4*b - a = 20 :=
  sorry

end value_of_4b_minus_a_l1416_141632


namespace exists_such_h_l1416_141613

noncomputable def exists_h (h : ℝ) : Prop :=
  ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋)

theorem exists_such_h : ∃ h : ℝ, exists_h h := 
  -- Let's construct the h as mentioned in the provided proof
  ⟨1969^2 / 1968, 
    by sorry⟩

end exists_such_h_l1416_141613


namespace trevor_comic_first_issue_pages_l1416_141614

theorem trevor_comic_first_issue_pages
  (x : ℕ) 
  (h1 : 3 * x + 4 = 220) :
  x = 72 := 
by
  sorry

end trevor_comic_first_issue_pages_l1416_141614


namespace range_of_k_l1416_141629

theorem range_of_k (x : ℝ) (h1 : 0 < x) (h2 : x < 2) (h3 : x / Real.exp x < 1 / (k + 2 * x - x^2)) :
    0 ≤ k ∧ k < Real.exp 1 - 1 :=
sorry

end range_of_k_l1416_141629


namespace exists_point_at_distance_l1416_141669

def Line : Type := sorry
def Point : Type := sorry
def distance (P Q : Point) : ℝ := sorry

variables (L : Line) (d : ℝ) (P : Point)

def is_at_distance (Q : Point) (L : Line) (d : ℝ) := ∃ Q, distance Q L = d

theorem exists_point_at_distance :
  ∃ Q : Point, is_at_distance Q L d :=
sorry

end exists_point_at_distance_l1416_141669


namespace complement_of_angle_correct_l1416_141648

def complement_of_angle (a : ℚ) : ℚ := 90 - a

theorem complement_of_angle_correct : complement_of_angle (40 + 30/60) = 49 + 30/60 :=
by
  -- placeholder for the proof
  sorry

end complement_of_angle_correct_l1416_141648


namespace LCM_is_4199_l1416_141696

theorem LCM_is_4199 :
  let beats_of_cymbals := 13
  let beats_of_triangle := 17
  let beats_of_tambourine := 19
  Nat.lcm (Nat.lcm beats_of_cymbals beats_of_triangle) beats_of_tambourine = 4199 := 
by 
  sorry 

end LCM_is_4199_l1416_141696


namespace midpoint_sum_four_times_l1416_141663

theorem midpoint_sum_four_times (x1 y1 x2 y2 : ℝ) (h1 : x1 = 8) (h2 : y1 = -4) (h3 : x2 = -2) (h4 : y2 = 10) :
  4 * ((x1 + x2) / 2 + (y1 + y2) / 2) = 24 :=
by
  rw [h1, h2, h3, h4]
  -- simplifying to get the desired result
  sorry

end midpoint_sum_four_times_l1416_141663


namespace ellipse_foci_on_x_axis_l1416_141624

variable {a b : ℝ}

theorem ellipse_foci_on_x_axis (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1) (hc : ∀ x y : ℝ, (a * x^2 + b * y^2 = 1) → (1 / a > 1 / b ∧ 1 / b > 0))
  : 0 < a ∧ a < b :=
sorry

end ellipse_foci_on_x_axis_l1416_141624


namespace remove_remaining_wallpaper_time_l1416_141695

noncomputable def time_per_wall : ℕ := 2
noncomputable def walls_dining_room : ℕ := 4
noncomputable def walls_living_room : ℕ := 4
noncomputable def walls_completed : ℕ := 1

theorem remove_remaining_wallpaper_time : 
    time_per_wall * (walls_dining_room - walls_completed) + time_per_wall * walls_living_room = 14 :=
by
  sorry

end remove_remaining_wallpaper_time_l1416_141695


namespace five_letters_three_mailboxes_l1416_141672

theorem five_letters_three_mailboxes : (∃ n : ℕ, n = 5) ∧ (∃ m : ℕ, m = 3) → ∃ k : ℕ, k = m^n :=
by
  sorry

end five_letters_three_mailboxes_l1416_141672


namespace average_test_score_first_25_percent_l1416_141685

theorem average_test_score_first_25_percent (x : ℝ) :
  (0.25 * x) + (0.50 * 65) + (0.25 * 90) = 1 * 75 → x = 80 :=
by
  sorry

end average_test_score_first_25_percent_l1416_141685


namespace solve_star_eq_l1416_141650

noncomputable def star (a b : ℤ) : ℤ := if a = b then 2 else sorry

axiom star_assoc : ∀ (a b c : ℤ), star a (star b c) = (star a b) - c
axiom star_self_eq_two : ∀ (a : ℤ), star a a = 2

theorem solve_star_eq : ∀ (x : ℤ), star 100 (star 5 x) = 20 → x = 20 :=
by
  intro x hx
  sorry

end solve_star_eq_l1416_141650


namespace terminating_decimal_zeros_l1416_141645

-- Define a generic environment for terminating decimal and problem statement
def count_zeros (d : ℚ) : ℕ :=
  -- This function needs to count the zeros after the decimal point and before
  -- the first non-zero digit, but its actual implementation is skipped here.
  sorry

-- Define the specific fraction in question
def my_fraction : ℚ := 1 / (2^3 * 5^5)

-- State what we need to prove: the number of zeros after the decimal point
-- in the terminating representation of my_fraction should be 4
theorem terminating_decimal_zeros : count_zeros my_fraction = 4 :=
by
  -- Proof is skipped
  sorry

end terminating_decimal_zeros_l1416_141645


namespace three_four_five_six_solution_l1416_141622

-- State that the equation 3^x + 4^x = 5^x is true when x=2
axiom three_four_five_solution : 3^2 + 4^2 = 5^2

-- We need to prove the following theorem
theorem three_four_five_six_solution : 3^3 + 4^3 + 5^3 = 6^3 :=
by sorry

end three_four_five_six_solution_l1416_141622


namespace f_of_2_l1416_141690

def f (x : ℝ) : ℝ := sorry

theorem f_of_2 : f 2 = 20 / 3 :=
    sorry

end f_of_2_l1416_141690


namespace no_real_solution_l1416_141604

theorem no_real_solution (x : ℝ) : ¬ (x^3 + 2 * (x + 1)^3 + 3 * (x + 2)^3 = 6 * (x + 4)^3) :=
sorry

end no_real_solution_l1416_141604


namespace average_speed_round_trip_l1416_141617

theorem average_speed_round_trip (D T : ℝ) (h1 : D = 51 * T) : (2 * D) / (3 * T) = 34 := 
by
  sorry

end average_speed_round_trip_l1416_141617


namespace product_of_squares_l1416_141621

theorem product_of_squares (x : ℝ) (h : |5 * x| + 4 = 49) : x^2 * (if x = 9 then 9 else -9)^2 = 6561 :=
by
  sorry

end product_of_squares_l1416_141621


namespace square_tablecloth_side_length_l1416_141618

theorem square_tablecloth_side_length (area : ℝ) (h : area = 5) : ∃ a : ℝ, a > 0 ∧ a * a = 5 := 
by
  use Real.sqrt 5
  constructor
  · apply Real.sqrt_pos.2; linarith
  · exact Real.mul_self_sqrt (by linarith [h])

end square_tablecloth_side_length_l1416_141618


namespace charity_fundraising_l1416_141603

theorem charity_fundraising (num_people : ℕ) (amount_event1 amount_event2 : ℕ) (total_amount_per_person : ℕ) :
  num_people = 8 →
  amount_event1 = 2000 →
  amount_event2 = 1000 →
  total_amount_per_person = (amount_event1 + amount_event2) / num_people →
  total_amount_per_person = 375 :=
by
  intros h1 h2 h3 h4
  sorry

end charity_fundraising_l1416_141603


namespace measure_four_messzely_l1416_141683

theorem measure_four_messzely (c3 c5 : ℕ) (hc3 : c3 = 3) (hc5 : c5 = 5) : 
  ∃ (x y z : ℕ), x = 4 ∧ x + y * c3 + z * c5 = 4 := 
sorry

end measure_four_messzely_l1416_141683


namespace find_OH_squared_l1416_141674

variables (A B C : ℝ) (a b c R OH : ℝ)

-- Conditions
def circumcenter (O : ℝ) := true  -- Placeholder, as the actual definition relies on geometric properties
def orthocenter (H : ℝ) := true   -- Placeholder, as the actual definition relies on geometric properties

axiom eqR : R = 5
axiom sumSquares : a^2 + b^2 + c^2 = 50

-- Problem statement
theorem find_OH_squared : OH^2 = 175 :=
by
  sorry

end find_OH_squared_l1416_141674


namespace predicted_value_y_at_x_5_l1416_141602

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem predicted_value_y_at_x_5 :
  let x_values := [-2, -1, 0, 1, 2]
  let y_values := [5, 4, 2, 2, 1]
  let x_bar := mean x_values
  let y_bar := mean y_values
  let a_hat := y_bar
  (∀ x, y = -x + a_hat) →
  (x = 5 → y = -2.2) :=
by
  sorry

end predicted_value_y_at_x_5_l1416_141602


namespace product_of_roots_l1416_141653

theorem product_of_roots : ∀ x : ℝ, (x + 3) * (x - 4) = 17 → (∃ a b : ℝ, (x = a ∨ x = b) ∧ a * b = -29) :=
by
  sorry

end product_of_roots_l1416_141653


namespace evaluation_of_expression_l1416_141637

theorem evaluation_of_expression :
  10 * (1 / 8) - 6.4 / 8 + 1.2 * 0.125 = 0.6 :=
by sorry

end evaluation_of_expression_l1416_141637


namespace monthly_interest_payment_l1416_141694

theorem monthly_interest_payment (principal : ℝ) (annual_rate : ℝ) (months_in_year : ℝ) : 
  principal = 31200 → 
  annual_rate = 0.09 → 
  months_in_year = 12 → 
  (principal * annual_rate) / months_in_year = 234 := 
by 
  intros h_principal h_rate h_months
  rw [h_principal, h_rate, h_months]
  sorry

end monthly_interest_payment_l1416_141694


namespace find_x_given_inverse_relationship_l1416_141649

theorem find_x_given_inverse_relationship :
  ∀ (x y: ℝ), (0 < x ∧ 0 < y) ∧ ((x^3 * y = 64) ↔ (x = 2 ∧ y = 8)) ∧ (y = 500) →
  x = 2 / 5 :=
by
  intros x y h
  sorry

end find_x_given_inverse_relationship_l1416_141649


namespace weekly_charge_for_motel_l1416_141631

theorem weekly_charge_for_motel (W : ℝ) (h1 : ∀ t : ℝ, t = 3 * 4 → t = 12)
(h2 : ∀ cost_weekly : ℝ, cost_weekly = 12 * W)
(h3 : ∀ cost_monthly : ℝ, cost_monthly = 3 * 1000)
(h4 : cost_monthly + 360 = 12 * W) : 
W = 280 := 
sorry

end weekly_charge_for_motel_l1416_141631


namespace complex_number_solution_l1416_141610

open Complex

theorem complex_number_solution (z : ℂ) (h : (2 * z - I) * (2 - I) = 5) : 
  z = 1 + I :=
sorry

end complex_number_solution_l1416_141610


namespace diamond_comm_l1416_141639

def diamond (a b : ℝ) : ℝ := a^2 * b^2 - a^2 - b^2

theorem diamond_comm (x y : ℝ) : diamond x y = diamond y x := by
  sorry

end diamond_comm_l1416_141639


namespace adjust_collection_amount_l1416_141638

/-- Define the error caused by mistaking half-dollars for dollars -/
def halfDollarError (x : ℕ) : ℤ := 50 * x

/-- Define the error caused by mistaking quarters for nickels -/
def quarterError (x : ℕ) : ℤ := 20 * x

/-- Define the total error based on the given conditions -/
def totalError (x : ℕ) : ℤ := halfDollarError x - quarterError x

theorem adjust_collection_amount (x : ℕ) : totalError x = 30 * x := by
  sorry

end adjust_collection_amount_l1416_141638


namespace find_d_l1416_141654

theorem find_d (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 = c * (d + 20)) (h2 : b^2 = c * (d - 18)) :
  d = 180 :=
sorry

end find_d_l1416_141654


namespace nicholas_bottle_caps_l1416_141605

theorem nicholas_bottle_caps (N : ℕ) (h : N + 85 = 93) : N = 8 :=
by
  sorry

end nicholas_bottle_caps_l1416_141605


namespace percentage_of_green_eyed_brunettes_l1416_141635

def conditions (a b c d : ℝ) : Prop :=
  (a / (a + b) = 0.65) ∧
  (b / (b + c) = 0.7) ∧
  (c / (c + d) = 0.1)

theorem percentage_of_green_eyed_brunettes (a b c d : ℝ) (h : conditions a b c d) :
  d / (a + b + c + d) = 0.54 :=
sorry

end percentage_of_green_eyed_brunettes_l1416_141635


namespace max_value_x_minus_y_proof_l1416_141615

noncomputable def max_value_x_minus_y (θ : ℝ) : ℝ :=
  sorry

theorem max_value_x_minus_y_proof (θ : ℝ) (h1 : x = Real.sin θ) (h2 : y = Real.cos θ)
(h3 : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) (h4 : (x^2 + y^2)^2 = x + y) : 
  max_value_x_minus_y θ = Real.sqrt 2 :=
sorry

end max_value_x_minus_y_proof_l1416_141615


namespace cube_root_of_neg_eight_l1416_141657

theorem cube_root_of_neg_eight : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by
  sorry

end cube_root_of_neg_eight_l1416_141657


namespace each_boy_receives_52_l1416_141664

theorem each_boy_receives_52 {boys girls : ℕ} (h_ratio : boys / gcd boys girls = 5 ∧ girls / gcd boys girls = 7) (h_total : boys + girls = 180) (h_share : 3900 ∣ boys) :
  3900 / boys = 52 :=
by
  sorry

end each_boy_receives_52_l1416_141664


namespace chess_probability_l1416_141679

theorem chess_probability (P_draw P_B_win : ℚ) (h_draw : P_draw = 1/2) (h_B_win : P_B_win = 1/3) :
  (1 - P_draw - P_B_win = 1/6) ∧ -- Statement A is correct
  (P_draw + (1 - P_draw - P_B_win) ≠ 1/2) ∧ -- Statement B is incorrect as it's not 1/2
  (1 - P_draw - P_B_win ≠ 2/3) ∧ -- Statement C is incorrect as it's not 2/3
  (P_draw + P_B_win ≠ 1/2) := -- Statement D is incorrect as it's not 1/2
by
  -- Insert proof here
  sorry

end chess_probability_l1416_141679


namespace arithmetic_sequence_sum_l1416_141665

theorem arithmetic_sequence_sum (d : ℕ) (y : ℕ) (x : ℕ) (h_y : y = 39) (h_d : d = 6) 
  (h_x : x = y - d) : 
  x + y = 72 := by 
  sorry

end arithmetic_sequence_sum_l1416_141665
