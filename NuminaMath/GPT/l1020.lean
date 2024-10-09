import Mathlib

namespace geometric_sequence_sum_l1020_102009

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (r : ℝ)
  (h1 : a 1 + a 3 = 8)
  (h2 : a 5 + a 7 = 4)
  (geometric_seq : ∀ n, a n = a 1 * r ^ (n - 1)) :
  a 9 + a 11 + a 13 + a 15 = 3 :=
by
  sorry

end geometric_sequence_sum_l1020_102009


namespace log_base_9_of_729_l1020_102055

theorem log_base_9_of_729 : ∃ x : ℝ, (9:ℝ) = 3^2 ∧ (729:ℝ) = 3^6 ∧ (9:ℝ)^x = 729 ∧ x = 3 :=
by
  sorry

end log_base_9_of_729_l1020_102055


namespace product_of_two_numbers_l1020_102017

theorem product_of_two_numbers (x y : ℝ) (h_diff : x - y = 12) (h_sum_of_squares : x^2 + y^2 = 245) : x * y = 50.30 :=
sorry

end product_of_two_numbers_l1020_102017


namespace number_is_12_l1020_102066

theorem number_is_12 (x : ℝ) (h : 4 * x - 3 = 9 * (x - 7)) : x = 12 :=
by
  sorry

end number_is_12_l1020_102066


namespace polynomial_equivalence_l1020_102084

-- Define the polynomial T in terms of x.
def T (x : ℝ) : ℝ := (x-2)^5 + 5 * (x-2)^4 + 10 * (x-2)^3 + 10 * (x-2)^2 + 5 * (x-2) + 1

-- Define the target polynomial.
def target (x : ℝ) : ℝ := (x-1)^5

-- State the theorem that T is equivalent to target.
theorem polynomial_equivalence (x : ℝ) : T x = target x :=
by
  sorry

end polynomial_equivalence_l1020_102084


namespace kicks_before_break_l1020_102075

def total_kicks : ℕ := 98
def kicks_after_break : ℕ := 36
def kicks_needed_to_goal : ℕ := 19

theorem kicks_before_break :
  total_kicks - (kicks_after_break + kicks_needed_to_goal) = 43 := 
by
  -- proof wanted
  sorry

end kicks_before_break_l1020_102075


namespace calc_x_squared_plus_5xy_plus_y_squared_l1020_102007

theorem calc_x_squared_plus_5xy_plus_y_squared 
  (x y : ℝ) 
  (h1 : x * y = 4)
  (h2 : x - y = 5) :
  x^2 + 5 * x * y + y^2 = 53 :=
by 
  sorry

end calc_x_squared_plus_5xy_plus_y_squared_l1020_102007


namespace bowling_team_scores_l1020_102048

theorem bowling_team_scores : 
  ∀ (A B C : ℕ), 
  C = 162 → 
  B = 3 * C → 
  A + B + C = 810 → 
  A / B = 1 / 3 := 
by 
  intros A B C h1 h2 h3 
  sorry

end bowling_team_scores_l1020_102048


namespace work_days_A_l1020_102051

theorem work_days_A (x : ℝ) (h1 : ∀ y : ℝ, y = 20) (h2 : ∀ z : ℝ, z = 5) 
  (h3 : ∀ w : ℝ, w = 0.41666666666666663) :
  x = 15 :=
  sorry

end work_days_A_l1020_102051


namespace find_a3_l1020_102058

noncomputable def S (n : ℕ) (a₁ q : ℚ) : ℚ :=
  a₁ * (1 - q ^ n) / (1 - q)

noncomputable def a (n : ℕ) (a₁ q : ℚ) : ℚ :=
  a₁ * q ^ (n - 1)

theorem find_a3 (a₁ q : ℚ) (h1 : S 6 a₁ q / S 3 a₁ q = -19 / 8)
  (h2 : a 4 a₁ q - a 2 a₁ q = -15 / 8) :
  a 3 a₁ q = 9 / 4 :=
by sorry

end find_a3_l1020_102058


namespace sixteen_grams_on_left_pan_l1020_102038

theorem sixteen_grams_on_left_pan :
  ∃ (weights : ℕ → ℕ) (pans : ℕ → ℕ) (n : ℕ),
    weights n = 16 ∧
    pans 0 = 11111 ∧
    ∃ k, (∀ i < k, weights i = 2 ^ i) ∧
    (∀ i < k, (pans 1 + weights i = 38) ∧ (pans 0 + 11111 = weights i + skeletal)) ∧
    k = 6 := by
  sorry

end sixteen_grams_on_left_pan_l1020_102038


namespace positive_diff_solutions_l1020_102042

theorem positive_diff_solutions (x1 x2 : ℝ) (h1 : 2 * x1 - 3 = 14) (h2 : 2 * x2 - 3 = -14) : 
  x1 - x2 = 14 := 
by
  sorry

end positive_diff_solutions_l1020_102042


namespace pounds_per_ton_l1020_102049

theorem pounds_per_ton (packet_count : ℕ) (packet_weight_pounds : ℚ) (packet_weight_ounces : ℚ) (ounces_per_pound : ℚ) (total_weight_tons : ℚ) (total_weight_pounds : ℚ) :
  packet_count = 1760 →
  packet_weight_pounds = 16 →
  packet_weight_ounces = 4 →
  ounces_per_pound = 16 →
  total_weight_tons = 13 →
  total_weight_pounds = (packet_count * (packet_weight_pounds + (packet_weight_ounces / ounces_per_pound))) →
  total_weight_pounds / total_weight_tons = 2200 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end pounds_per_ton_l1020_102049


namespace find_product_in_geometric_sequence_l1020_102073

def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) / a n = a 1 / a 0

theorem find_product_in_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 7 * a 13 = 8) : 
  a 3 * a 11 = 4 :=
by sorry

end find_product_in_geometric_sequence_l1020_102073


namespace sum_of_angles_l1020_102069

theorem sum_of_angles (A B C x y : ℝ) 
  (hA : A = 34) 
  (hB : B = 80) 
  (hC : C = 30)
  (pentagon_angles_sum : A + B + (360 - x) + 90 + (120 - y) = 540) : 
  x + y = 144 :=
by
  sorry

end sum_of_angles_l1020_102069


namespace infinite_solutions_exists_l1020_102047

theorem infinite_solutions_exists :
  ∃ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x ≠ y) ∧ (y ≠ z) ∧ (z ≠ x) ∧
  (x - y + z = 1) ∧ ((x * y) % z = 0) ∧ ((y * z) % x = 0) ∧ ((z * x) % y = 0) ∧
  ∀ n : ℕ, ∃ x y z : ℕ, (n > 0) ∧ (x = n * (n^2 + n - 1)) ∧ (y = (n+1) * (n^2 + n - 1)) ∧ (z = n * (n+1)) := by
  sorry

end infinite_solutions_exists_l1020_102047


namespace claire_photos_l1020_102014

-- Define the number of photos taken by Claire, Lisa, and Robert
variables (C L R : ℕ)

-- Conditions based on the problem
def Lisa_photos (C : ℕ) := 3 * C
def Robert_photos (C : ℕ) := C + 24

-- Prove that C = 12 given the conditions
theorem claire_photos : 
  (L = Lisa_photos C) ∧ (R = Robert_photos C) ∧ (L = R) → C = 12 := 
by
  sorry

end claire_photos_l1020_102014


namespace total_children_l1020_102019

-- Given the conditions
def toy_cars : Nat := 134
def dolls : Nat := 269

-- Prove that the total number of children is 403
theorem total_children (h_cars : toy_cars = 134) (h_dolls : dolls = 269) :
  toy_cars + dolls = 403 :=
by
  sorry

end total_children_l1020_102019


namespace abs_sum_ge_sqrt_three_over_two_l1020_102031

open Real

theorem abs_sum_ge_sqrt_three_over_two
  (a b : ℝ) : (|a| + |b| ≥ 2 / sqrt 3) ∧ (∀ x, |a * sin x + b * sin (2 * x)| ≤ 1) ↔
  (a, b) = (4 / (3 * sqrt 3), 2 / (3 * sqrt 3)) ∨ 
  (a, b) = (-4 / (3 * sqrt 3), -2 / (3 * sqrt 3)) ∨
  (a, b) = (4 / (3 * sqrt 3), -2 / (3 * sqrt 3)) ∨
  (a, b) = (-4 / (3 * sqrt 3), 2 / (3 * sqrt 3)) := 
sorry

end abs_sum_ge_sqrt_three_over_two_l1020_102031


namespace buffaloes_added_l1020_102052

-- Let B be the daily fodder consumption of one buffalo in units
noncomputable def daily_fodder_buffalo (B : ℝ) := B
noncomputable def daily_fodder_cow (B : ℝ) := (3 / 4) * B
noncomputable def daily_fodder_ox (B : ℝ) := (3 / 2) * B

-- Initial conditions
def initial_buffaloes := 15
def initial_cows := 24
def initial_oxen := 8
def initial_days := 24
noncomputable def total_initial_fodder (B : ℝ) := (initial_buffaloes * daily_fodder_buffalo B) + (initial_oxen * daily_fodder_ox B) + (initial_cows * daily_fodder_cow B)
noncomputable def total_fodder (B : ℝ) := total_initial_fodder B * initial_days

-- New conditions after adding cows and buffaloes
def additional_cows := 60
def new_days := 9
noncomputable def total_new_daily_fodder (B : ℝ) (x : ℝ) := ((initial_buffaloes + x) * daily_fodder_buffalo B) + (initial_oxen * daily_fodder_ox B) + ((initial_cows + additional_cows) * daily_fodder_cow B)

-- Proof statement: Prove that given the conditions, the number of additional buffaloes, x, is 30.
theorem buffaloes_added (B : ℝ) : 
  (total_fodder B = total_new_daily_fodder B 30 * new_days) :=
by sorry

end buffaloes_added_l1020_102052


namespace cos_double_angle_l1020_102025

theorem cos_double_angle (α : ℝ) (h : Real.sin (π / 2 - α) = 1 / 4) : 
  Real.cos (2 * α) = -7 / 8 :=
sorry

end cos_double_angle_l1020_102025


namespace possible_integer_roots_l1020_102032

def polynomial (x : ℤ) : ℤ := x^3 + 2 * x^2 - 3 * x - 17

theorem possible_integer_roots :
  ∃ (roots : List ℤ), roots = [1, -1, 17, -17] ∧ ∀ r ∈ roots, polynomial r = 0 := 
sorry

end possible_integer_roots_l1020_102032


namespace sin_45_eq_sqrt2_div_2_l1020_102081

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l1020_102081


namespace melissa_trip_total_time_l1020_102030

theorem melissa_trip_total_time :
  ∀ (freeway_dist rural_dist : ℕ) (freeway_speed_factor : ℕ) 
  (rural_time : ℕ),
  freeway_dist = 80 →
  rural_dist = 20 →
  freeway_speed_factor = 4 →
  rural_time = 40 →
  (rural_dist * freeway_speed_factor / rural_time + freeway_dist / (rural_dist * freeway_speed_factor / rural_time)) = 80 :=
by
  intros freeway_dist rural_dist freeway_speed_factor rural_time hd1 hd2 hd3 hd4
  sorry

end melissa_trip_total_time_l1020_102030


namespace negation_of_p_equiv_l1020_102086

-- Define the initial proposition p
def p : Prop := ∃ x : ℝ, x^2 - 5*x - 6 < 0

-- State the theorem for the negation of p
theorem negation_of_p_equiv : ¬p ↔ ∀ x : ℝ, x^2 - 5*x - 6 ≥ 0 :=
by
  sorry

end negation_of_p_equiv_l1020_102086


namespace represent_in_scientific_notation_l1020_102096

def million : ℕ := 10^6
def rural_residents : ℝ := 42.39 * million

theorem represent_in_scientific_notation :
  42.39 * 10^6 = 4.239 * 10^7 :=
by
  -- The proof is omitted.
  sorry

end represent_in_scientific_notation_l1020_102096


namespace find_difference_l1020_102043

theorem find_difference (a b : ℕ) (h1 : a < b) (h2 : a + b = 78) (h3 : Nat.lcm a b = 252) : b - a = 6 :=
by
  sorry

end find_difference_l1020_102043


namespace tina_wins_before_first_loss_l1020_102097

-- Definitions based on conditions
variable (W : ℕ) -- The number of wins before Tina's first loss

-- Conditions
def win_before_first_loss : W = 10 := by sorry

def total_wins (W : ℕ) := W + 2 * W -- After her first loss, she doubles her wins and loses again
def total_losses : ℕ := 2 -- She loses twice

def career_record_condition (W : ℕ) : Prop := total_wins W - total_losses = 28

-- Proof Problem (Statement)
theorem tina_wins_before_first_loss : career_record_condition W → W = 10 :=
by sorry

end tina_wins_before_first_loss_l1020_102097


namespace parametric_to_general_eq_l1020_102057

theorem parametric_to_general_eq (x y θ : ℝ) 
  (h1 : x = 2 + Real.sin θ ^ 2) 
  (h2 : y = -1 + Real.cos (2 * θ)) : 
  2 * x + y - 4 = 0 ∧ 2 ≤ x ∧ x ≤ 3 := 
sorry

end parametric_to_general_eq_l1020_102057


namespace average_infect_influence_l1020_102008

theorem average_infect_influence
  (x : ℝ)
  (h : (1 + x)^2 = 100) :
  x = 9 :=
sorry

end average_infect_influence_l1020_102008


namespace sqrt_subtraction_l1020_102080

theorem sqrt_subtraction :
  (Real.sqrt (49 + 81)) - (Real.sqrt (36 - 9)) = (Real.sqrt 130) - (3 * Real.sqrt 3) :=
sorry

end sqrt_subtraction_l1020_102080


namespace inverse_function_of_1_div_2_pow_eq_log_base_1_div_2_l1020_102079

noncomputable def inverse_of_half_pow (x : ℝ) : ℝ := Real.log x / Real.log (1 / 2)

theorem inverse_function_of_1_div_2_pow_eq_log_base_1_div_2 (x : ℝ) (hx : 0 < x) :
  inverse_of_half_pow x = Real.log x / Real.log (1 / 2) :=
by
  sorry

end inverse_function_of_1_div_2_pow_eq_log_base_1_div_2_l1020_102079


namespace badminton_tournament_l1020_102046

theorem badminton_tournament (n x : ℕ) (h1 : 2 * n > 0) (h2 : 3 * n > 0) (h3 : (5 * n) * (5 * n - 1) = 14 * x) : n = 3 :=
by
  -- Placeholder for the proof
  sorry

end badminton_tournament_l1020_102046


namespace weights_divide_three_piles_l1020_102090

theorem weights_divide_three_piles (n : ℕ) (h : n > 3) :
  (∃ (k : ℕ), n = 3 * k ∨ n = 3 * k + 2) ↔
  (∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range (n + 1) ∧
   A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
   A.sum id = (n * (n + 1)) / 6 ∧ B.sum id = (n * (n + 1)) / 6 ∧ C.sum id = (n * (n + 1)) / 6) :=
sorry

end weights_divide_three_piles_l1020_102090


namespace intercepts_of_line_l1020_102021

theorem intercepts_of_line (x y : ℝ) 
  (h : 2 * x + 7 * y = 35) :
  (y = 5 → x = 0) ∧ (x = 17.5 → y = 0)  :=
by
  sorry

end intercepts_of_line_l1020_102021


namespace geometric_series_sum_l1020_102023

-- Definitions based on conditions
def a : ℚ := 3 / 2
def r : ℚ := -4 / 9

-- Statement of the proof
theorem geometric_series_sum : (a / (1 - r)) = 27 / 26 :=
by
  -- proof goes here
  sorry

end geometric_series_sum_l1020_102023


namespace fraction_given_to_friend_l1020_102013

theorem fraction_given_to_friend (s u r g k : ℕ) 
  (h1: s = 135) 
  (h2: u = s / 3) 
  (h3: r = s - u) 
  (h4: k = 54) 
  (h5: g = r - k) :
  g / r = 2 / 5 := 
  by
  sorry

end fraction_given_to_friend_l1020_102013


namespace find_fraction_value_l1020_102063

variable (a b : ℝ)
variable (h1 : b > a)
variable (h2 : a > 0)
variable (h3 : a / b + b / a = 4)

theorem find_fraction_value (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a / b + b / a = 4) : (a + b) / (a - b) = Real.sqrt 3 := by
  sorry

end find_fraction_value_l1020_102063


namespace sum_of_remainders_mod_13_l1020_102099

theorem sum_of_remainders_mod_13 :
  ∀ (a b c d e : ℤ),
    a ≡ 3 [ZMOD 13] →
    b ≡ 5 [ZMOD 13] →
    c ≡ 7 [ZMOD 13] →
    d ≡ 9 [ZMOD 13] →
    e ≡ 11 [ZMOD 13] →
    (a + b + c + d + e) % 13 = 9 :=
by
  intros a b c d e ha hb hc hd he
  sorry

end sum_of_remainders_mod_13_l1020_102099


namespace fraction_identity_l1020_102024

theorem fraction_identity (a b c : ℝ) (h1 : a + b + c > 0) (h2 : a + b - c > 0) (h3 : a + c - b > 0) (h4 : b + c - a > 0) 
  (h5 : (a+b+c)/(a+b-c) = 7) (h6 : (a+b+c)/(a+c-b) = 1.75) : (a+b+c)/(b+c-a) = 3.5 :=
by
  sorry

end fraction_identity_l1020_102024


namespace solution_of_inequality_system_l1020_102059

theorem solution_of_inequality_system (a b : ℝ) 
    (h1 : 4 - 2 * a = 0)
    (h2 : (3 + b) / 2 = 1) : a + b = 1 := 
by 
  sorry

end solution_of_inequality_system_l1020_102059


namespace ryan_chinese_learning_hours_l1020_102087

variable (hours_english : ℕ)
variable (days : ℕ)
variable (total_hours : ℕ)

theorem ryan_chinese_learning_hours (h1 : hours_english = 6) 
                                    (h2 : days = 5) 
                                    (h3 : total_hours = 65) : 
                                    total_hours - (hours_english * days) / days = 7 := by
  sorry

end ryan_chinese_learning_hours_l1020_102087


namespace find_divisor_l1020_102040

theorem find_divisor (D Q R d: ℕ) (hD: D = 16698) (hQ: Q = 89) (hR: R = 14) (hDiv: D = d * Q + R): d = 187 := 
by 
  sorry

end find_divisor_l1020_102040


namespace value_of_M_l1020_102068

theorem value_of_M (M : ℝ) (h : 0.2 * M = 500) : M = 2500 :=
by
  sorry

end value_of_M_l1020_102068


namespace rectangle_perimeter_l1020_102088

theorem rectangle_perimeter (breadth length : ℝ) (h1 : length = 3 * breadth) (h2 : length * breadth = 147) : 2 * length + 2 * breadth = 56 :=
by
  sorry

end rectangle_perimeter_l1020_102088


namespace find_f_at_2_l1020_102006

def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x - 8

theorem find_f_at_2 (a b c : ℝ) (h : f (-2) a b c = 10) : f 2 a b c = -26 :=
by
  sorry

end find_f_at_2_l1020_102006


namespace sin_315_degree_l1020_102041

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l1020_102041


namespace heloise_gives_dogs_to_janet_l1020_102015

theorem heloise_gives_dogs_to_janet :
  ∃ d c : ℕ, d * 17 = c * 10 ∧ d + c = 189 ∧ d - 60 = 10 :=
by
  sorry

end heloise_gives_dogs_to_janet_l1020_102015


namespace probability_all_three_blue_l1020_102026

theorem probability_all_three_blue :
  let total_jellybeans := 20
  let initial_blue := 10
  let initial_red := 10
  let prob_first_blue := initial_blue / total_jellybeans
  let prob_second_blue := (initial_blue - 1) / (total_jellybeans - 1)
  let prob_third_blue := (initial_blue - 2) / (total_jellybeans - 2)
  prob_first_blue * prob_second_blue * prob_third_blue = 2 / 19 := 
by
  sorry

end probability_all_three_blue_l1020_102026


namespace power_ineq_for_n_geq_5_l1020_102050

noncomputable def power_ineq (n : ℕ) : Prop := 2^n > n^2 + 1

theorem power_ineq_for_n_geq_5 (n : ℕ) (h : n ≥ 5) : power_ineq n :=
  sorry

end power_ineq_for_n_geq_5_l1020_102050


namespace pieces_count_l1020_102002

def pieces_after_n_tears (n : ℕ) : ℕ :=
  3 * n + 1

theorem pieces_count (n : ℕ) : pieces_after_n_tears n = 3 * n + 1 :=
by
  sorry

end pieces_count_l1020_102002


namespace chess_sequences_l1020_102020

def binomial (n k : Nat) : Nat := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem chess_sequences :
  binomial 11 4 = 210 := by
  sorry

end chess_sequences_l1020_102020


namespace focus_on_negative_y_axis_l1020_102004

-- Definition of the condition: equation of the parabola
def parabola (x y : ℝ) := x^2 + y = 0

-- Statement of the problem
theorem focus_on_negative_y_axis (x y : ℝ) (h : parabola x y) : 
  -- The focus of the parabola lies on the negative half of the y-axis
  ∃ y, y < 0 :=
sorry

end focus_on_negative_y_axis_l1020_102004


namespace girls_in_school_play_l1020_102005

theorem girls_in_school_play (G : ℕ) (boys : ℕ) (total_parents : ℕ)
  (h1 : boys = 8) (h2 : total_parents = 28) (h3 : 2 * boys + 2 * G = total_parents) : 
  G = 6 :=
sorry

end girls_in_school_play_l1020_102005


namespace prob_of_B1_selected_prob_of_D1_in_team_l1020_102074

noncomputable def total_teams : ℕ := 20

noncomputable def teams_with_B1 : ℕ := 8

noncomputable def teams_with_D1 : ℕ := 12

theorem prob_of_B1_selected : (teams_with_B1 : ℚ) / total_teams = 2 / 5 := by
  sorry

theorem prob_of_D1_in_team : (teams_with_D1 : ℚ) / total_teams = 3 / 5 := by
  sorry

end prob_of_B1_selected_prob_of_D1_in_team_l1020_102074


namespace polygon_divided_into_7_triangles_l1020_102089

theorem polygon_divided_into_7_triangles (n : ℕ) (h : n - 2 = 7) : n = 9 :=
by
  sorry

end polygon_divided_into_7_triangles_l1020_102089


namespace find_polynomials_l1020_102065

-- Define our polynomial P(x)
def polynomial_condition (P : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, (x-1) * P.eval (x+1) - (x+2) * P.eval x = 0

-- State the theorem
theorem find_polynomials (P : Polynomial ℝ) :
  polynomial_condition P ↔ ∃ a : ℝ, P = Polynomial.C a * (Polynomial.X^3 - Polynomial.X) :=
by
  sorry

end find_polynomials_l1020_102065


namespace sticker_distribution_probability_l1020_102094

theorem sticker_distribution_probability :
  let p := 32
  let q := 50050
  p + q = 50082 :=
sorry

end sticker_distribution_probability_l1020_102094


namespace spherical_coords_standard_form_l1020_102011

theorem spherical_coords_standard_form :
  ∀ (ρ θ φ : ℝ), ρ > 0 → 0 ≤ θ ∧ θ < 2 * Real.pi → 0 ≤ φ ∧ φ ≤ Real.pi →
  (5, (5 * Real.pi) / 7, (11 * Real.pi) / 6) = (ρ, θ, φ) →
  (ρ, (12 * Real.pi) / 7, Real.pi / 6) = (ρ, θ, φ) :=
by 
  intros ρ θ φ hρ hθ hφ h_eq
  sorry

end spherical_coords_standard_form_l1020_102011


namespace triangle_inequality_internal_point_l1020_102070

theorem triangle_inequality_internal_point {A B C P : Type} 
  (x y z p q r : ℝ) 
  (h_distances_from_vertices : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_distances_from_sides : p > 0 ∧ q > 0 ∧ r > 0)
  (h_x_y_z_triangle_ineq : x + y > z ∧ y + z > x ∧ z + x > y)
  (h_p_q_r_triangle_ineq : p + q > r ∧ q + r > p ∧ r + p > q) :
  x * y * z ≥ (q + r) * (r + p) * (p + q) :=
sorry

end triangle_inequality_internal_point_l1020_102070


namespace train_crossing_time_l1020_102036

noncomputable def train_length : ℝ := 385
noncomputable def train_speed_kmph : ℝ := 90
noncomputable def bridge_length : ℝ := 1250

noncomputable def convert_speed_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def time_to_cross_bridge (train_length bridge_length train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_mps := convert_speed_to_mps train_speed_kmph
  total_distance / speed_mps

theorem train_crossing_time :
  time_to_cross_bridge train_length bridge_length train_speed_kmph = 65.4 :=
by
  sorry

end train_crossing_time_l1020_102036


namespace cube_volume_multiple_of_6_l1020_102056

theorem cube_volume_multiple_of_6 (n : ℕ) (h : ∃ m : ℕ, n^3 = 24 * m) : ∃ k : ℕ, n = 6 * k :=
by
  sorry

end cube_volume_multiple_of_6_l1020_102056


namespace sin_double_angle_l1020_102016

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l1020_102016


namespace real_return_l1020_102093

theorem real_return (n i r: ℝ) (h₁ : n = 0.21) (h₂ : i = 0.10) : 
  (1 + r) = (1 + n) / (1 + i) → r = 0.10 :=
by
  intro h₃
  sorry

end real_return_l1020_102093


namespace proof_problem_l1020_102083

open Real

def p : Prop := ∀ a : ℝ, a^2017 > -1 → a > -1
def q : Prop := ∀ x : ℝ, x^2 * tan (x^2) > 0

theorem proof_problem : p ∨ q :=
sorry

end proof_problem_l1020_102083


namespace minimum_value_y_is_2_l1020_102010

noncomputable def minimum_value_y (x : ℝ) : ℝ :=
  x + (1 / x)

theorem minimum_value_y_is_2 (x : ℝ) (hx : 0 < x) : 
  (∀ y, y = minimum_value_y x → y ≥ 2) :=
by
  sorry

end minimum_value_y_is_2_l1020_102010


namespace erica_total_earnings_l1020_102085

def fishPrice : Nat := 20
def pastCatch : Nat := 80
def todayCatch : Nat := 2 * pastCatch
def pastEarnings := pastCatch * fishPrice
def todayEarnings := todayCatch * fishPrice
def totalEarnings := pastEarnings + todayEarnings

theorem erica_total_earnings : totalEarnings = 4800 := by
  sorry

end erica_total_earnings_l1020_102085


namespace direction_cosines_l1020_102054

theorem direction_cosines (x y z : ℝ) (α β γ : ℝ)
  (h1 : 2 * x - 3 * y - 3 * z - 9 = 0)
  (h2 : x - 2 * y + z + 3 = 0) :
  α = 9 / Real.sqrt 107 ∧ β = 5 / Real.sqrt 107 ∧ γ = 1 / Real.sqrt 107 :=
by
  -- Here, we will sketch out the proof to establish that these values for α, β, and γ hold.
  sorry

end direction_cosines_l1020_102054


namespace orchids_initially_three_l1020_102091

-- Define initial number of roses and provided number of orchids in the vase
def initial_roses : ℕ := 9
def added_orchids (O : ℕ) : ℕ := 13
def added_roses : ℕ := 3
def difference := 10

-- Define initial number of orchids that we need to prove
def initial_orchids (O : ℕ) : Prop :=
  added_orchids O - added_roses = difference →
  O = 3

theorem orchids_initially_three :
  initial_orchids O :=
sorry

end orchids_initially_three_l1020_102091


namespace power_calculation_l1020_102064

theorem power_calculation (a : ℝ) (m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 5) : a^(3*m + 2*n) = 200 := by
  sorry

end power_calculation_l1020_102064


namespace max_three_cell_corners_l1020_102076

-- Define the grid size
def grid_height : ℕ := 7
def grid_width : ℕ := 14

-- Define the concept of a three-cell corner removal
def three_cell_corner (region : ℕ) : ℕ := region / 3

-- Define the problem statement in Lean
theorem max_three_cell_corners : three_cell_corner (grid_height * grid_width) = 32 := by
  sorry

end max_three_cell_corners_l1020_102076


namespace ratio_of_a_to_c_l1020_102029

variable {a b c d : ℚ}

theorem ratio_of_a_to_c (h₁ : a / b = 5 / 4) (h₂ : c / d = 4 / 3) (h₃ : d / b = 1 / 5) : 
  a / c = 75 / 16 := 
sorry

end ratio_of_a_to_c_l1020_102029


namespace beam_reflection_equation_l1020_102037

theorem beam_reflection_equation:
  ∃ (line : ℝ → ℝ → Prop), 
  (∀ (x y : ℝ), line x y ↔ (5 * x - 2 * y - 10 = 0)) ∧
  (line 4 5) ∧ 
  (line 2 0) :=
by
  sorry

end beam_reflection_equation_l1020_102037


namespace jerry_age_l1020_102027

theorem jerry_age (M J : ℝ) (h₁ : M = 17) (h₂ : M = 2.5 * J - 3) : J = 8 :=
by
  -- The proof is omitted.
  sorry

end jerry_age_l1020_102027


namespace altitude_segment_length_l1020_102045

theorem altitude_segment_length 
  {A B C D E : Type} 
  (BD DC AE y : ℝ) 
  (h1 : BD = 4) 
  (h2 : DC = 6) 
  (h3 : AE = 3) 
  (h4 : 3 / 4 = 9 / (y + 3)) : 
  y = 9 := 
by 
  sorry

end altitude_segment_length_l1020_102045


namespace Fred_last_week_l1020_102092

-- Definitions from conditions
def Fred_now := 40
def Fred_earned := 21

-- The theorem we need to prove
theorem Fred_last_week :
  Fred_now - Fred_earned = 19 :=
by
  sorry

end Fred_last_week_l1020_102092


namespace cubes_with_4_neighbors_l1020_102028

theorem cubes_with_4_neighbors (a b c : ℕ) (h₁ : 3 < a) (h₂ : 3 < b) (h₃ : 3 < c)
  (h₄ : (a - 2) * (b - 2) * (c - 2) = 429) : 
  4 * ((a - 2) + (b - 2) + (c - 2)) = 108 := by
  sorry

end cubes_with_4_neighbors_l1020_102028


namespace homes_termite_ridden_but_not_collapsing_fraction_l1020_102098

variable (H : Type) -- Representing Homes on Gotham Street

def termite_ridden_fraction : ℚ := 1 / 3
def collapsing_fraction_given_termite_ridden : ℚ := 7 / 10

theorem homes_termite_ridden_but_not_collapsing_fraction :
  (termite_ridden_fraction * (1 - collapsing_fraction_given_termite_ridden)) = 1 / 10 :=
by
  sorry

end homes_termite_ridden_but_not_collapsing_fraction_l1020_102098


namespace carson_total_distance_l1020_102035

def perimeter (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length + width)

def total_distance (length : ℕ) (width : ℕ) (rounds : ℕ) (breaks : ℕ) (break_distance : ℕ) : ℕ :=
  let P := perimeter length width
  let distance_rounds := rounds * P
  let distance_breaks := breaks * break_distance
  distance_rounds + distance_breaks

theorem carson_total_distance :
  total_distance 600 400 8 4 100 = 16400 :=
by
  sorry

end carson_total_distance_l1020_102035


namespace find_n_l1020_102077

open Nat

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given condition for the proof
def condition (n : ℕ) : Prop := binom (n + 1) 7 - binom n 7 = binom n 8

-- The statement to prove
theorem find_n (n : ℕ) (h : condition n) : n = 14 :=
sorry

end find_n_l1020_102077


namespace tan_ratio_l1020_102001

-- Definitions of the problem conditions
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides opposite to the angles

-- The given equation condition
axiom h : a * Real.cos B - b * Real.cos A = (4 / 5) * c

-- The goal is to prove the value of tan(A) / tan(B)
theorem tan_ratio (A B C : ℝ) (a b c : ℝ) (h : a * Real.cos B - b * Real.cos A = (4 / 5) * c) :
  Real.tan A / Real.tan B = 9 :=
sorry

end tan_ratio_l1020_102001


namespace total_action_figures_l1020_102072

-- Definitions based on conditions
def initial_figures : ℕ := 8
def figures_per_set : ℕ := 5
def added_sets : ℕ := 2
def total_added_figures : ℕ := added_sets * figures_per_set
def total_figures : ℕ := initial_figures + total_added_figures

-- Theorem statement with conditions and expected result
theorem total_action_figures : total_figures = 18 := by
  sorry

end total_action_figures_l1020_102072


namespace sqrt_x_plus_5_l1020_102022

theorem sqrt_x_plus_5 (x : ℝ) (h : x = -1) : Real.sqrt (x + 5) = 2 :=
by
  sorry

end sqrt_x_plus_5_l1020_102022


namespace geometric_sequence_sum_l1020_102033

-- Defining the geometric sequence related properties and conditions
theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * r) → 
  S 3 = a 0 + a 1 + a 2 →
  S 6 = a 3 + a 4 + a 5 →
  S 12 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 →
  S 3 = 3 →
  S 6 = 6 →
  S 12 = 45 :=
by
  sorry

end geometric_sequence_sum_l1020_102033


namespace proof_of_area_weighted_sum_of_distances_l1020_102095

def area_weighted_sum_of_distances
  (a b a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) 
  (t1 t2 t3 t4 t : ℝ)
  (z1 z2 z3 z4 z : ℝ) 
  (h1 : t1 = a1 * b1)
  (h2 : t2 = (a - a1) * b1)
  (h3 : t3 = a3 * b3)
  (h4 : t4 = (a - a3) * b3)
  (rect_area : t = a * b)
  : Prop :=
  t1 * z1 + t2 * z2 + t3 * z3 + t4 * z4 = t * z

theorem proof_of_area_weighted_sum_of_distances
  (a b a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (t1 t2 t3 t4 t : ℝ)
  (z1 z2 z3 z4 z : ℝ)
  (h1 : t1 = a1 * b1)
  (h2 : t2 = (a - a1) * b1)
  (h3 : t3 = a3 * b3)
  (h4 : t4 = (a - a3) * b3)
  (rect_area : t = a * b)
  : area_weighted_sum_of_distances a b a1 a2 a3 a4 b1 b2 b3 b4 t1 t2 t3 t4 t z1 z2 z3 z4 z h1 h2 h3 h4 rect_area :=
  sorry

end proof_of_area_weighted_sum_of_distances_l1020_102095


namespace min_cost_29_disks_l1020_102071

theorem min_cost_29_disks
  (price_single : ℕ := 20) 
  (price_pack_10 : ℕ := 111) 
  (price_pack_25 : ℕ := 265) :
  ∃ cost : ℕ, cost ≥ (price_pack_10 + price_pack_10 + price_pack_10) 
              ∧ cost ≤ (price_pack_25 + price_single * 4) 
              ∧ cost = 333 := 
by
  sorry

end min_cost_29_disks_l1020_102071


namespace quadratic_greatest_value_and_real_roots_l1020_102044

theorem quadratic_greatest_value_and_real_roots :
  (∀ x : ℝ, -x^2 + 9 * x - 20 ≥ 0 → x ≤ 5)
  ∧ (∃ x : ℝ, -x^2 + 9 * x - 20 = 0)
  :=
sorry

end quadratic_greatest_value_and_real_roots_l1020_102044


namespace simplify_exponents_l1020_102078

variable (x : ℝ)

theorem simplify_exponents (x : ℝ) : (x^5) * (x^2) = x^(7) :=
by
  sorry

end simplify_exponents_l1020_102078


namespace minimum_value_xy_l1020_102012

theorem minimum_value_xy (x y : ℝ) (h : (x + Real.sqrt (x^2 + 1)) * (y + Real.sqrt (y^2 + 1)) ≥ 1) : x + y ≥ 0 :=
sorry

end minimum_value_xy_l1020_102012


namespace vacation_total_cost_l1020_102053

def plane_ticket_cost (per_person_cost : ℕ) (num_people : ℕ) : ℕ :=
  num_people * per_person_cost

def hotel_stay_cost (per_person_per_day_cost : ℕ) (num_people : ℕ) (num_days : ℕ) : ℕ :=
  num_people * per_person_per_day_cost * num_days

def total_vacation_cost (plane_ticket_cost : ℕ) (hotel_stay_cost : ℕ) : ℕ :=
  plane_ticket_cost + hotel_stay_cost

theorem vacation_total_cost :
  let per_person_plane_ticket_cost := 24
  let per_person_hotel_cost := 12
  let num_people := 2
  let num_days := 3
  let plane_cost := plane_ticket_cost per_person_plane_ticket_cost num_people
  let hotel_cost := hotel_stay_cost per_person_hotel_cost num_people num_days
  total_vacation_cost plane_cost hotel_cost = 120 := by
  sorry

end vacation_total_cost_l1020_102053


namespace remainder_7_pow_253_mod_12_l1020_102039

theorem remainder_7_pow_253_mod_12 : (7 ^ 253) % 12 = 7 := by
  sorry

end remainder_7_pow_253_mod_12_l1020_102039


namespace number_of_newborn_members_l1020_102018

theorem number_of_newborn_members (N : ℝ) (h : (9/10 : ℝ) ^ 3 * N = 291.6) : N = 400 :=
sorry

end number_of_newborn_members_l1020_102018


namespace terminal_side_in_quadrant_l1020_102000

theorem terminal_side_in_quadrant (k : ℤ) (α : ℝ)
  (h: π + 2 * k * π < α ∧ α < (3 / 2) * π + 2 * k * π) :
  (π / 2) + k * π < α / 2 ∧ α / 2 < (3 / 4) * π + k * π :=
sorry

end terminal_side_in_quadrant_l1020_102000


namespace simplify_and_evaluate_expression_l1020_102034

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l1020_102034


namespace first_year_after_2022_with_digit_sum_5_l1020_102061

def sum_of_digits (n : ℕ) : ℕ :=
  (toString n).foldl (λ acc c => acc + c.toNat - '0'.toNat) 0

theorem first_year_after_2022_with_digit_sum_5 :
  ∃ y : ℕ, y > 2022 ∧ sum_of_digits y = 5 ∧ ∀ z : ℕ, z > 2022 ∧ z < y → sum_of_digits z ≠ 5 :=
sorry

end first_year_after_2022_with_digit_sum_5_l1020_102061


namespace class_b_students_l1020_102003

theorem class_b_students (total_students : ℕ) (sample_size : ℕ) (class_a_sample : ℕ) :
  total_students = 100 → sample_size = 10 → class_a_sample = 4 → 
  (total_students - total_students * class_a_sample / sample_size = 60) :=
by
  intros
  sorry

end class_b_students_l1020_102003


namespace equilateral_triangle_area_l1020_102082

theorem equilateral_triangle_area (A B C P : ℝ × ℝ)
  (hABC : ∃ a b c : ℝ, a = b ∧ b = c ∧ a = dist A B ∧ b = dist B C ∧ c = dist C A)
  (hPA : dist P A = 10)
  (hPB : dist P B = 8)
  (hPC : dist P C = 12) :
  ∃ (area : ℝ), area = 104 :=
by
  sorry

end equilateral_triangle_area_l1020_102082


namespace inequality_with_conditions_l1020_102060

variable {a b c : ℝ}

theorem inequality_with_conditions (h : a * b + b * c + c * a = 1) :
  (|a - b| / |1 + c^2|) + (|b - c| / |1 + a^2|) ≥ (|c - a| / |1 + b^2|) :=
by
  sorry

end inequality_with_conditions_l1020_102060


namespace marie_age_l1020_102067

theorem marie_age (L M O : ℕ) (h1 : L = 4 * M) (h2 : O = M + 8) (h3 : L = O) : M = 8 / 3 := by
  sorry

end marie_age_l1020_102067


namespace hotel_P_charge_less_than_G_l1020_102062

open Real

variable (G R P : ℝ)

-- Given conditions
def charge_R_eq_2G : Prop := R = 2 * G
def charge_P_eq_R_minus_55percent : Prop := P = R - 0.55 * R

-- Goal: Prove the percentage by which P's charge is less than G's charge is 10%
theorem hotel_P_charge_less_than_G : charge_R_eq_2G G R → charge_P_eq_R_minus_55percent R P → P = 0.9 * G := by
  intros h1 h2
  sorry

end hotel_P_charge_less_than_G_l1020_102062
