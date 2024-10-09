import Mathlib

namespace find_original_price_l1514_151456

theorem find_original_price (reduced_price : ℝ) (percent : ℝ) (original_price : ℝ) 
  (h1 : reduced_price = 6) (h2 : percent = 0.25) (h3 : reduced_price = percent * original_price) : 
  original_price = 24 :=
sorry

end find_original_price_l1514_151456


namespace cricket_initial_overs_l1514_151499

theorem cricket_initial_overs
  (x : ℕ)
  (hx1 : ∃ x : ℕ, 0 ≤ x)
  (initial_run_rate : ℝ)
  (remaining_run_rate : ℝ)
  (remaining_overs : ℕ)
  (target_runs : ℕ)
  (H1 : initial_run_rate = 3.2)
  (H2 : remaining_run_rate = 6.25)
  (H3 : remaining_overs = 40)
  (H4 : target_runs = 282) :
  3.2 * (x : ℝ) + 6.25 * 40 = 282 → x = 10 := 
by 
  simp only [H1, H2, H3, H4]
  sorry

end cricket_initial_overs_l1514_151499


namespace part1_part2_l1514_151468

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + 1

theorem part1 :
  ∀ x : ℝ, f (x + π) = f x :=
by sorry

theorem part2 :
  ∃ (max_x min_x : ℝ), max_x ∈ Set.Icc (π/12) (π/4) ∧ min_x ∈ Set.Icc (π/12) (π/4) ∧
    f max_x = 7 / 4 ∧ f min_x = (5 + Real.sqrt 3) / 4 ∧
    (max_x = π / 6) ∧ (min_x = π / 12 ∨ min_x = π / 4) :=
by sorry

end part1_part2_l1514_151468


namespace archery_competition_l1514_151469

theorem archery_competition (points : Finset ℕ) (product : ℕ) : 
  points = {11, 7, 5, 2} ∧ product = 38500 → 
  ∃ n : ℕ, n = 7 := 
by
  intros h
  sorry

end archery_competition_l1514_151469


namespace Mabel_marble_count_l1514_151429

variable (K A M : ℕ)

axiom Amanda_condition : A + 12 = 2 * K
axiom Mabel_K_condition : M = 5 * K
axiom Mabel_A_condition : M = A + 63

theorem Mabel_marble_count : M = 85 := by
  sorry

end Mabel_marble_count_l1514_151429


namespace emily_sixth_quiz_score_l1514_151478

theorem emily_sixth_quiz_score (a1 a2 a3 a4 a5 : ℕ) (target_mean : ℕ) (sixth_score : ℕ) :
  a1 = 94 ∧ a2 = 97 ∧ a3 = 88 ∧ a4 = 90 ∧ a5 = 102 ∧ target_mean = 95 →
  sixth_score = (target_mean * 6 - (a1 + a2 + a3 + a4 + a5)) →
  sixth_score = 99 :=
by
  sorry

end emily_sixth_quiz_score_l1514_151478


namespace ratio_to_percent_l1514_151498

theorem ratio_to_percent (a b : ℕ) (h : a = 6) (h2 : b = 3) :
  ((a / b : ℚ) * 100 = 200) :=
by
  have h3 : a = 6 := h
  have h4 : b = 3 := h2
  sorry

end ratio_to_percent_l1514_151498


namespace JulioHasMoreSoda_l1514_151476

-- Define the number of bottles each person has
def JulioOrangeBottles : ℕ := 4
def JulioGrapeBottles : ℕ := 7
def MateoOrangeBottles : ℕ := 1
def MateoGrapeBottles : ℕ := 3

-- Define the volume of each bottle in liters
def BottleVolume : ℕ := 2

-- Define the total liters of soda each person has
def JulioTotalLiters : ℕ := JulioOrangeBottles * BottleVolume + JulioGrapeBottles * BottleVolume
def MateoTotalLiters : ℕ := MateoOrangeBottles * BottleVolume + MateoGrapeBottles * BottleVolume

-- Prove the difference in total liters of soda between Julio and Mateo
theorem JulioHasMoreSoda : JulioTotalLiters - MateoTotalLiters = 14 := by
  sorry

end JulioHasMoreSoda_l1514_151476


namespace compute_expression_l1514_151471

theorem compute_expression : 12 * (1 / 17) * 34 = 24 :=
by sorry

end compute_expression_l1514_151471


namespace translate_quadratic_vertex_right_l1514_151406

theorem translate_quadratic_vertex_right : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = 2 * (x - 4)^2 - 3) ∧ 
  (∃ (g : ℝ → ℝ), (∀ x, g x = 2 * ((x - 1) - 3)^2 - 3))) → 
  (∃ v : ℝ × ℝ, v = (4, -3)) :=
sorry

end translate_quadratic_vertex_right_l1514_151406


namespace pipes_fill_tank_in_10_hours_l1514_151466

noncomputable def R_A := 1 / 70
noncomputable def R_B := 2 * R_A
noncomputable def R_C := 2 * R_B
noncomputable def R_total := R_A + R_B + R_C
noncomputable def T := 1 / R_total

theorem pipes_fill_tank_in_10_hours :
  T = 10 := 
sorry

end pipes_fill_tank_in_10_hours_l1514_151466


namespace find_value_b_l1514_151426

-- Define the problem-specific elements
noncomputable def is_line_eqn (y x : ℝ) : Prop := y = 4 - 2 * x

theorem find_value_b (b : ℝ) (h₀ : b > 0) (h₁ : b < 2)
  (hP : ∀ y, is_line_eqn y 0 → y = 4)
  (hS : ∀ y, is_line_eqn y 2 → y = 0)
  (h_ratio : ∀ Q R S O P,
    Q = (2, 0) ∧ R = (2, 0) ∧ S = (2, 0) ∧ P = (0, 4) ∧ O = (0, 0) →
    4 / 9 = 4 / ((Q.1 - O.1) * (Q.1 - O.1)) →
    (Q.1 - O.1) / (P.2 - O.2) = 2 / 3) :
  b = 2 :=
sorry

end find_value_b_l1514_151426


namespace x_squared_minus_y_squared_l1514_151485

theorem x_squared_minus_y_squared (x y : ℚ) (h₁ : x + y = 9 / 17) (h₂ : x - y = 1 / 51) : x^2 - y^2 = 1 / 289 :=
by
  sorry

end x_squared_minus_y_squared_l1514_151485


namespace number_with_1_before_and_after_l1514_151462

theorem number_with_1_before_and_after (n : ℕ) (hn : n < 10) : 100 * 1 + 10 * n + 1 = 101 + 10 * n := by
    sorry

end number_with_1_before_and_after_l1514_151462


namespace pure_imaginary_z_squared_l1514_151402

-- Formalization in Lean 4
theorem pure_imaginary_z_squared (a : ℝ) (h : a + (1 + a) * I = (1 + a) * I) : (a + (1 + a) * I)^2 = -1 :=
by
  sorry

end pure_imaginary_z_squared_l1514_151402


namespace num_triangles_with_perimeter_20_l1514_151445

theorem num_triangles_with_perimeter_20 : 
  ∃ (triangles : List (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ triangles → a + b + c = 20 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ b + c > a ∧ c + a > b) ∧ 
    triangles.length = 8 :=
sorry

end num_triangles_with_perimeter_20_l1514_151445


namespace range_of_a_for_decreasing_function_l1514_151449

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x + 4 else 3 * a / x

theorem range_of_a_for_decreasing_function :
  (∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≥ f a x2) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end range_of_a_for_decreasing_function_l1514_151449


namespace minimum_value_of_f_l1514_151425

noncomputable def f (x : ℝ) : ℝ := x^2 + (1 / x^2) + (1 / (x^2 + 1 / x^2))

theorem minimum_value_of_f (x : ℝ) (hx : x > 0) : ∃ y : ℝ, y = f x ∧ y >= 5 / 2 :=
by
  sorry

end minimum_value_of_f_l1514_151425


namespace prob_TeamA_wins_2_1_proof_prob_TeamB_wins_proof_best_of_five_increases_prob_l1514_151494

noncomputable def prob_TeamA_wins_game : ℝ := 0.6
noncomputable def prob_TeamB_wins_game : ℝ := 0.4

-- Probability of Team A winning 2-1 in a best-of-three
noncomputable def prob_TeamA_wins_2_1 : ℝ := 2 * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game 

-- Probability of Team B winning in a best-of-three
noncomputable def prob_TeamB_wins_2_0 : ℝ := prob_TeamB_wins_game * prob_TeamB_wins_game
noncomputable def prob_TeamB_wins_2_1 : ℝ := 2 * prob_TeamB_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game
noncomputable def prob_TeamB_wins : ℝ := prob_TeamB_wins_2_0 + prob_TeamB_wins_2_1

-- Probability of Team A winning in a best-of-three
noncomputable def prob_TeamA_wins_best_of_three : ℝ := 1 - prob_TeamB_wins

-- Probability of Team A winning in a best-of-five
noncomputable def prob_TeamA_wins_3_0 : ℝ := prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamA_wins_game
noncomputable def prob_TeamA_wins_3_1 : ℝ := 3 * (prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game)
noncomputable def prob_TeamA_wins_3_2 : ℝ := 6 * (prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game)

noncomputable def prob_TeamA_wins_best_of_five : ℝ := prob_TeamA_wins_3_0 + prob_TeamA_wins_3_1 + prob_TeamA_wins_3_2

theorem prob_TeamA_wins_2_1_proof :
  prob_TeamA_wins_2_1 = 0.288 :=
sorry

theorem prob_TeamB_wins_proof :
  prob_TeamB_wins = 0.352 :=
sorry

theorem best_of_five_increases_prob :
  prob_TeamA_wins_best_of_three < prob_TeamA_wins_best_of_five :=
sorry

end prob_TeamA_wins_2_1_proof_prob_TeamB_wins_proof_best_of_five_increases_prob_l1514_151494


namespace geom_sequence_sum_l1514_151454

theorem geom_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n : ℕ, a n > 0)
  (h_geom : ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q)
  (h_eq : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) :
  a 5 + a 7 = 6 :=
sorry

end geom_sequence_sum_l1514_151454


namespace reflection_matrix_condition_l1514_151491

noncomputable def reflection_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, b], ![-(3/4 : ℝ), 1/4]]

noncomputable def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], ![0, 1]]

theorem reflection_matrix_condition (a b : ℝ) :
  (reflection_matrix a b)^2 = identity_matrix ↔ a = -(1/4) ∧ b = -(3/4) :=
  by
  sorry

end reflection_matrix_condition_l1514_151491


namespace min_tries_to_get_blue_and_yellow_l1514_151411

theorem min_tries_to_get_blue_and_yellow 
  (purple blue yellow : ℕ) 
  (h_purple : purple = 7) 
  (h_blue : blue = 5)
  (h_yellow : yellow = 11) :
  ∃ n, n = 9 ∧ (∀ tries, tries ≥ n → (∃ i j, (i ≤ purple ∧ j ≤ tries - i ∧ j ≤ blue) → (∃ k, k = tries - i - j ∧ k ≤ yellow))) :=
by sorry

end min_tries_to_get_blue_and_yellow_l1514_151411


namespace amount_b_l1514_151427

variable {a b : ℚ} -- a and b are rational numbers

theorem amount_b (h1 : a + b = 1210) (h2 : (4 / 15) * a = (2 / 5) * b) : b = 484 :=
sorry

end amount_b_l1514_151427


namespace find_value_of_A_l1514_151463

theorem find_value_of_A (A ω φ c : ℝ)
  (a : ℕ+ → ℝ)
  (h_seq : ∀ n : ℕ+, a n * a (n + 1) * a (n + 2) = a n + a (n + 1) + a (n + 2))
  (h_neq : ∀ n : ℕ+, a n * a (n + 1) ≠ 1)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 2)
  (h_form : ∀ n : ℕ+, a n = A * Real.sin (ω * n + φ) + c)
  (h_ω_gt_0 : ω > 0)
  (h_phi_lt_pi_div_2 : |φ| < Real.pi / 2) :
  A = -2 * Real.sqrt 3 / 3 := 
sorry

end find_value_of_A_l1514_151463


namespace Don_poured_milk_correct_amount_l1514_151486

theorem Don_poured_milk_correct_amount :
  let original_milk : ℚ := 3 / 8
  let portion_poured : ℚ := 5 / 6
  let poured_milk : ℚ := portion_poured * original_milk
  poured_milk = 5 / 16 :=
by
  let original_milk : ℚ := 3 / 8
  let portion_poured : ℚ := 5 / 6
  let poured_milk : ℚ := portion_poured * original_milk
  show poured_milk = 5 / 16
  sorry

end Don_poured_milk_correct_amount_l1514_151486


namespace smallest_perfect_square_divisible_by_5_and_7_l1514_151497

theorem smallest_perfect_square_divisible_by_5_and_7 
  (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n = k^2)
  (h3 : 5 ∣ n)
  (h4 : 7 ∣ n) : 
  n = 1225 :=
sorry

end smallest_perfect_square_divisible_by_5_and_7_l1514_151497


namespace problem_l1514_151409

theorem problem (p q r : ℝ) (h1 : p + q + r = 5000) (h2 : r = (2 / 3) * (p + q)) : r = 2000 :=
by
  sorry

end problem_l1514_151409


namespace amrita_bakes_cake_next_thursday_l1514_151404

theorem amrita_bakes_cake_next_thursday (n m : ℕ) (h1 : n = 5) (h2 : m = 7) : Nat.lcm n m = 35 :=
by
  -- Proof goes here
  sorry

end amrita_bakes_cake_next_thursday_l1514_151404


namespace azalea_wool_price_l1514_151424

noncomputable def sheep_count : ℕ := 200
noncomputable def wool_per_sheep : ℕ := 10
noncomputable def shearing_cost : ℝ := 2000
noncomputable def profit : ℝ := 38000

-- Defining total wool and total revenue based on these definitions
noncomputable def total_wool : ℕ := sheep_count * wool_per_sheep
noncomputable def total_revenue : ℝ := profit + shearing_cost
noncomputable def price_per_pound : ℝ := total_revenue / total_wool

-- Problem statement: Proving that the price per pound of wool is equal to $20
theorem azalea_wool_price :
  price_per_pound = 20 := 
sorry

end azalea_wool_price_l1514_151424


namespace greatest_perfect_power_sum_l1514_151446

def sum_c_d_less_500 : ℕ :=
  let c := 22
  let d := 2
  c + d

theorem greatest_perfect_power_sum :
  ∃ c d : ℕ, 0 < c ∧ 1 < d ∧ c^d < 500 ∧
  ∀ x y : ℕ, 0 < x ∧ 1 < y ∧ x^y < 500 → x^y ≤ c^d ∧ (c + d = 24) :=
by
  sorry

end greatest_perfect_power_sum_l1514_151446


namespace problem1_problem2_l1514_151467

-- Problem 1
theorem problem1 (a b : ℝ) : 
  a^2 * (2 * a * b - 1) + (a - 3 * b) * (a + b) = 2 * a^3 * b - 2 * a * b - 3 * b^2 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (2 * x - 3)^2 - (x + 2)^2 = 3 * x^2 - 16 * x + 5 :=
by sorry

end problem1_problem2_l1514_151467


namespace probability_two_yellow_apples_l1514_151418

theorem probability_two_yellow_apples (total_apples : ℕ) (red_apples : ℕ) (green_apples : ℕ) (yellow_apples : ℕ) (choose : ℕ → ℕ → ℕ) (probability : ℕ → ℕ → ℝ) :
  total_apples = 10 →
  red_apples = 5 →
  green_apples = 3 →
  yellow_apples = 2 →
  choose total_apples 2 = 45 →
  choose yellow_apples 2 = 1 →
  probability (choose yellow_apples 2) (choose total_apples 2) = 1 / 45 := 
  by
  sorry

end probability_two_yellow_apples_l1514_151418


namespace valid_odd_and_increasing_functions_l1514_151453

   def is_odd_function (f : ℝ → ℝ) : Prop :=
     ∀ x, f (-x) = -f (x)

   def is_increasing_function (f : ℝ → ℝ) : Prop :=
     ∀ x y, x < y → f (x) < f (y)

   noncomputable def f1 (x : ℝ) : ℝ := 3 * x^2
   noncomputable def f2 (x : ℝ) : ℝ := 6 * x
   noncomputable def f3 (x : ℝ) : ℝ := x * abs x
   noncomputable def f4 (x : ℝ) : ℝ := x + 1 / x

   theorem valid_odd_and_increasing_functions :
     (is_odd_function f2 ∧ is_increasing_function f2) ∧
     (is_odd_function f3 ∧ is_increasing_function f3) :=
   by
     sorry -- Proof goes here
   
end valid_odd_and_increasing_functions_l1514_151453


namespace monotonic_increasing_interval_l1514_151415

open Real

theorem monotonic_increasing_interval (k : ℤ) : 
  ∀ x, -π / 12 + k * π ≤ x ∧ x ≤ 5 * π / 12 + k * π ↔ 
    ∀ t, -π / 2 + 2 * k * π ≤ 2 * t - π / 3 ∧ 2 * t - π / 3 ≤ π / 2 + 2 * k * π :=
sorry

end monotonic_increasing_interval_l1514_151415


namespace sophia_age_in_three_years_l1514_151413

def current_age_jeremy : Nat := 40
def current_age_sebastian : Nat := current_age_jeremy + 4

def sum_ages_in_three_years (age_jeremy age_sebastian age_sophia : Nat) : Nat :=
  (age_jeremy + 3) + (age_sebastian + 3) + (age_sophia + 3)

theorem sophia_age_in_three_years (age_sophia : Nat) 
  (h1 : sum_ages_in_three_years current_age_jeremy current_age_sebastian age_sophia = 150) :
  age_sophia + 3 = 60 := by
  sorry

end sophia_age_in_three_years_l1514_151413


namespace range_of_m_l1514_151401

theorem range_of_m :
  (∀ x : ℝ, (x > 0) → (x^2 - m * x + 4 ≥ 0)) ∧ (¬∃ x : ℝ, (x^2 - 2 * m * x + 7 * m - 10 = 0)) ↔ (2 < m ∧ m ≤ 4) :=
by
  sorry

end range_of_m_l1514_151401


namespace farmer_initial_tomatoes_l1514_151488

theorem farmer_initial_tomatoes 
  (T : ℕ) -- The initial number of tomatoes
  (picked : ℕ)   -- The number of tomatoes picked
  (diff : ℕ) -- The difference between initial number of tomatoes and picked
  (h1 : picked = 9) -- The farmer picked 9 tomatoes
  (h2 : diff = 8) -- The difference is 8
  (h3 : T - picked = diff) -- T - 9 = 8
  :
  T = 17 := sorry

end farmer_initial_tomatoes_l1514_151488


namespace part1_part2_l1514_151440

theorem part1 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + 4*b^2 = 1/(a*b) + 3) :
  a*b ≤ 1 := sorry

theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + 4*b^2 = 1/(a*b) + 3) (hba : b > a) :
  1/a^3 - 1/b^3 ≥ 3 * (1/a - 1/b) := sorry

end part1_part2_l1514_151440


namespace number_of_2_dollar_socks_l1514_151405

theorem number_of_2_dollar_socks :
  ∃ (a b c : ℕ), (a + b + c = 15) ∧ (2 * a + 3 * b + 5 * c = 40) ∧ (a ≥ 1) ∧ (b ≥ 1) ∧ (c ≥ 1) ∧ (a = 7 ∨ a = 9 ∨ a = 11) :=
by {
  -- The details of the proof will go here, but we skip it for our requirements
  sorry
}

end number_of_2_dollar_socks_l1514_151405


namespace edward_remaining_money_l1514_151470

def initial_amount : ℕ := 19
def spent_amount : ℕ := 13
def remaining_amount : ℕ := initial_amount - spent_amount

theorem edward_remaining_money : remaining_amount = 6 := by
  sorry

end edward_remaining_money_l1514_151470


namespace sum_of_common_ratios_l1514_151448

-- Definitions for the geometric sequence conditions
def geom_seq_a (m : ℝ) (s : ℝ) (n : ℕ) : ℝ := m * s^n
def geom_seq_b (m : ℝ) (t : ℝ) (n : ℕ) : ℝ := m * t^n

-- Theorem statement
theorem sum_of_common_ratios (m s t : ℝ) (h₀ : m ≠ 0) (h₁ : s ≠ t) 
    (h₂ : geom_seq_a m s 2 - geom_seq_b m t 2 = 3 * (geom_seq_a m s 1 - geom_seq_b m t 1)) :
    s + t = 3 :=
by
  sorry

end sum_of_common_ratios_l1514_151448


namespace fair_attendance_l1514_151437

-- Define the variables x, y, and z
variables (x y z : ℕ)

-- Define the conditions given in the problem
def condition1 := z = 2 * y
def condition2 := x = z - 200
def condition3 := y = 600

-- State the main theorem proving the values of x, y, and z
theorem fair_attendance : condition1 y z → condition2 x z → condition3 y → (x = 1000 ∧ y = 600 ∧ z = 1200) := by
  intros h1 h2 h3
  sorry

end fair_attendance_l1514_151437


namespace nursing_home_received_boxes_l1514_151432

-- Each condition will be a definition in Lean 4.
def vitamins := 472
def supplements := 288
def total_boxes := 760

-- Statement of the proof problem in Lean
theorem nursing_home_received_boxes : vitamins + supplements = total_boxes := by
  sorry

end nursing_home_received_boxes_l1514_151432


namespace sufficient_but_not_necessary_l1514_151417

theorem sufficient_but_not_necessary (a : ℝ) :
  ((a + 2) * (3 * a - 4) - (a - 2) ^ 2 = 0 → a = 2 ∨ a = 1 / 2) →
  (a = 1 / 2 → ∃ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2) →
  ( (∀ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2 → a = 1/2) ∧ 
  (∃ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2) → a ≠ 1/2) :=
by
  sorry

end sufficient_but_not_necessary_l1514_151417


namespace managers_participation_l1514_151472

theorem managers_participation (teams : ℕ) (people_per_team : ℕ) (employees : ℕ) (total_people : teams * people_per_team = 6) (num_employees : employees = 3) :
  teams * people_per_team - employees = 3 :=
by
  sorry

end managers_participation_l1514_151472


namespace derivative_of_y_l1514_151421

open Real

noncomputable def y (x : ℝ) : ℝ := (cos (2 * x)) ^ ((log (cos (2 * x))) / 4)

theorem derivative_of_y (x : ℝ) :
  deriv y x = -((cos (2 * x)) ^ ((log (cos (2 * x))) / 4)) * (tan (2 * x)) * (log (cos (2 * x))) := by
    sorry

end derivative_of_y_l1514_151421


namespace sin_squared_plus_one_l1514_151464

theorem sin_squared_plus_one (x : ℝ) (hx : Real.tan x = 2) : Real.sin x ^ 2 + 1 = 9 / 5 := 
by 
  sorry

end sin_squared_plus_one_l1514_151464


namespace maximize_tetrahedron_volume_l1514_151459

noncomputable def volume_of_tetrahedron (a : ℝ) : ℝ :=
  a / 6

theorem maximize_tetrahedron_volume (a : ℝ) (h_a : 0 < a) 
  (P Q X Y : ℝ × ℝ × ℝ) (h_PQ : dist P Q = 1) (h_XY : dist X Y = 1) :
  volume_of_tetrahedron a = a / 6 :=
by
  sorry

end maximize_tetrahedron_volume_l1514_151459


namespace sum_of_squares_l1514_151422

theorem sum_of_squares (m n : ℝ) (h1 : m + n = 10) (h2 : m * n = 24) : m^2 + n^2 = 52 :=
by
  sorry

end sum_of_squares_l1514_151422


namespace arithmetic_sequence_length_l1514_151484

theorem arithmetic_sequence_length :
  ∃ n : ℕ, ∀ (a d l : ℕ), a = 2 → d = 5 → l = 3007 → l = a + (n-1) * d → n = 602 :=
by
  sorry

end arithmetic_sequence_length_l1514_151484


namespace distance_from_tangency_to_tangent_l1514_151473

theorem distance_from_tangency_to_tangent 
  (R r : ℝ)
  (hR : R = 3)
  (hr : r = 1)
  (externally_tangent : true) :
  ∃ d : ℝ, (d = 0 ∨ d = 7/3) :=
by
  sorry

end distance_from_tangency_to_tangent_l1514_151473


namespace tom_seashells_l1514_151420

theorem tom_seashells 
  (days_at_beach : ℕ) (seashells_per_day : ℕ) (total_seashells : ℕ) 
  (h1 : days_at_beach = 5) (h2 : seashells_per_day = 7) (h3 : total_seashells = days_at_beach * seashells_per_day) : 
  total_seashells = 35 := 
by
  rw [h1, h2] at h3 
  exact h3

end tom_seashells_l1514_151420


namespace base_8_sum_units_digit_l1514_151407

section
  def digit_in_base (n : ℕ) (base : ℕ) (d : ℕ) : Prop :=
  ((n % base) = d)

theorem base_8_sum_units_digit :
  let n1 := 63
  let n2 := 74
  let base := 8
  (digit_in_base n1 base 3) →
  (digit_in_base n2 base 4) →
  digit_in_base (n1 + n2) base 7 :=
by
  intro h1 h2
  -- placeholder for the detailed proof
  sorry
end

end base_8_sum_units_digit_l1514_151407


namespace probability_of_selection_of_Ram_l1514_151480

noncomputable def P_Ravi : ℚ := 1 / 5
noncomputable def P_Ram_and_Ravi : ℚ := 57 / 1000  -- This is the exact form of 0.05714285714285714

axiom independent_selection : ∀ (P_Ram P_Ravi : ℚ), P_Ram_and_Ravi = P_Ram * P_Ravi

theorem probability_of_selection_of_Ram (P_Ram : ℚ) : P_Ram_and_Ravi = P_Ram * P_Ravi → P_Ram = 2 / 7 := by
  intro h
  have h1 : P_Ram = P_Ram_and_Ravi / P_Ravi := sorry
  rw [h1, P_Ram_and_Ravi, P_Ravi]
  norm_num
  exact sorry

end probability_of_selection_of_Ram_l1514_151480


namespace pd_distance_l1514_151443

theorem pd_distance (PA PB PC PD : ℝ) (hPA : PA = 17) (hPB : PB = 15) (hPC : PC = 6) :
  PA^2 + PC^2 = PB^2 + PD^2 → PD = 10 :=
by
  sorry

end pd_distance_l1514_151443


namespace luke_fish_fillets_l1514_151450

def fish_per_day : ℕ := 2
def days : ℕ := 30
def fillets_per_fish : ℕ := 2

theorem luke_fish_fillets : fish_per_day * days * fillets_per_fish = 120 := 
by
  sorry

end luke_fish_fillets_l1514_151450


namespace count_lattice_points_on_hyperbola_l1514_151410

theorem count_lattice_points_on_hyperbola : 
  (∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), p ∈ S ↔ (p.1 ^ 2 - p.2 ^ 2 = 1800 ^ 2)) ∧ S.card = 150) :=
sorry

end count_lattice_points_on_hyperbola_l1514_151410


namespace arithmetic_expression_evaluation_l1514_151477

theorem arithmetic_expression_evaluation : 
  -6 * 3 - (-8 * -2) + (-7 * -5) - 10 = -9 := 
by
  sorry

end arithmetic_expression_evaluation_l1514_151477


namespace smallest_positive_integer_expr_2010m_44000n_l1514_151489

theorem smallest_positive_integer_expr_2010m_44000n :
  ∃ (m n : ℤ), 10 = gcd 2010 44000 :=
by
  sorry

end smallest_positive_integer_expr_2010m_44000n_l1514_151489


namespace sum_not_prime_l1514_151412

-- Definitions based on conditions:
variables {a b c d : ℕ}

-- Conditions:
axiom h_ab_eq_cd : a * b = c * d

-- Statement to prove:
theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : ¬Nat.Prime (a + b + c + d) :=
sorry

end sum_not_prime_l1514_151412


namespace find_k_l1514_151479

theorem find_k (k : ℕ) (h : 2 * 3 - k + 1 = 0) : k = 7 :=
sorry

end find_k_l1514_151479


namespace part1_solution_set_part2_values_of_a_l1514_151438

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l1514_151438


namespace mans_rate_in_still_water_l1514_151483

theorem mans_rate_in_still_water (R S : ℝ) (h1 : R + S = 18) (h2 : R - S = 4) : R = 11 :=
by {
  sorry
}

end mans_rate_in_still_water_l1514_151483


namespace number_of_tacos_l1514_151452

-- Define the conditions and prove the statement
theorem number_of_tacos (T : ℕ) :
  (4 * 7 + 9 * T = 37) → T = 1 :=
by
  intro h
  sorry

end number_of_tacos_l1514_151452


namespace nina_spends_70_l1514_151447

-- Definitions of the quantities and prices
def toys := 3
def toy_price := 10
def basketball_cards := 2
def card_price := 5
def shirts := 5
def shirt_price := 6

-- Calculate the total amount spent
def total_spent := (toys * toy_price) + (basketball_cards * card_price) + (shirts * shirt_price)

-- Problem statement: Prove that the total amount spent is $70
theorem nina_spends_70 : total_spent = 70 := by
  sorry

end nina_spends_70_l1514_151447


namespace solve_quadratic_eq_l1514_151435

theorem solve_quadratic_eq {x : ℝ} :
  (x = 3 ∨ x = -1) ↔ x^2 - 2 * x - 3 = 0 :=
by
  sorry

end solve_quadratic_eq_l1514_151435


namespace cafeteria_extra_fruits_l1514_151423

def red_apples_ordered : ℕ := 43
def green_apples_ordered : ℕ := 32
def oranges_ordered : ℕ := 25
def red_apples_chosen : ℕ := 7
def green_apples_chosen : ℕ := 5
def oranges_chosen : ℕ := 4

def extra_red_apples : ℕ := red_apples_ordered - red_apples_chosen
def extra_green_apples : ℕ := green_apples_ordered - green_apples_chosen
def extra_oranges : ℕ := oranges_ordered - oranges_chosen

def total_extra_fruits : ℕ := extra_red_apples + extra_green_apples + extra_oranges

theorem cafeteria_extra_fruits : total_extra_fruits = 84 := by
  sorry

end cafeteria_extra_fruits_l1514_151423


namespace simplify_expression_l1514_151461

theorem simplify_expression : -Real.sqrt 4 + abs (Real.sqrt 2 - 2) - 2023^0 = -2 := 
by 
  sorry

end simplify_expression_l1514_151461


namespace specified_percentage_of_number_is_40_l1514_151416

theorem specified_percentage_of_number_is_40 
  (N : ℝ) 
  (hN : (1 / 4) * (1 / 3) * (2 / 5) * N = 25) 
  (P : ℝ) 
  (hP : (P / 100) * N = 300) : 
  P = 40 := 
sorry

end specified_percentage_of_number_is_40_l1514_151416


namespace radius_large_circle_l1514_151490

/-- Definitions for the problem context -/
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

noncomputable def tangent_circles (c1 c2 : Circle) : Prop :=
dist c1.center c2.center = c1.radius + c2.radius

/-- Theorem to prove the radius of the large circle -/
theorem radius_large_circle 
  (small_circle : Circle)
  (h_radius : small_circle.radius = 2)
  (large_circle : Circle)
  (h_tangency1 : tangent_circles small_circle large_circle)
  (small_circle2 : Circle)
  (small_circle3 : Circle)
  (h_tangency2 : tangent_circles small_circle small_circle2)
  (h_tangency3 : tangent_circles small_circle small_circle3)
  (h_tangency4 : tangent_circles small_circle2 large_circle)
  (h_tangency5 : tangent_circles small_circle3 large_circle)
  (h_tangency6 : tangent_circles small_circle2 small_circle3)
  : large_circle.radius = 2 * (Real.sqrt 3 + 1) :=
sorry

end radius_large_circle_l1514_151490


namespace strawberries_per_box_l1514_151439

-- Define the initial conditions
def initial_strawberries : ℕ := 42
def additional_strawberries : ℕ := 78
def number_of_boxes : ℕ := 6

-- Define the total strawberries based on the given conditions
def total_strawberries : ℕ := initial_strawberries + additional_strawberries

-- The theorem to prove the number of strawberries per box
theorem strawberries_per_box : total_strawberries / number_of_boxes = 20 :=
by
  -- Proof steps would go here, but we use sorry since it's not required
  sorry

end strawberries_per_box_l1514_151439


namespace find_other_number_l1514_151444

open Nat

theorem find_other_number (A B lcm hcf : ℕ) (h_lcm : lcm = 2310) (h_hcf : hcf = 30) (h_A : A = 231) (h_eq : lcm * hcf = A * B) : 
  B = 300 :=
  sorry

end find_other_number_l1514_151444


namespace single_burger_cost_l1514_151451

-- Conditions
def total_cost : ℝ := 74.50
def total_burgers : ℕ := 50
def cost_double_burger : ℝ := 1.50
def double_burgers : ℕ := 49

-- Derived information
def cost_single_burger : ℝ := total_cost - (double_burgers * cost_double_burger)

-- Theorem: Prove the cost of a single burger
theorem single_burger_cost : cost_single_burger = 1.00 :=
by
  -- Proof goes here
  sorry

end single_burger_cost_l1514_151451


namespace flag_movement_distance_l1514_151455

theorem flag_movement_distance 
  (flagpole_length : ℝ)
  (half_mast : ℝ)
  (top_to_halfmast : ℝ)
  (halfmast_to_top : ℝ)
  (top_to_bottom : ℝ)
  (H1 : flagpole_length = 60)
  (H2 : half_mast = flagpole_length / 2)
  (H3 : top_to_halfmast = half_mast)
  (H4 : halfmast_to_top = half_mast)
  (H5 : top_to_bottom = flagpole_length) :
  top_to_halfmast + halfmast_to_top + top_to_halfmast + top_to_bottom = 180 := 
sorry

end flag_movement_distance_l1514_151455


namespace rectangle_width_length_ratio_l1514_151475

theorem rectangle_width_length_ratio (w l P : ℕ) (h_l : l = 10) (h_P : P = 30) (h_perimeter : 2*w + 2*l = P) :
  w / l = 1 / 2 := 
by {
  sorry
}

end rectangle_width_length_ratio_l1514_151475


namespace marcus_savings_l1514_151403

def MarcusMaxPrice : ℝ := 130
def ShoeInitialPrice : ℝ := 120
def DiscountPercentage : ℝ := 0.30
def FinalPrice : ℝ := ShoeInitialPrice - (DiscountPercentage * ShoeInitialPrice)
def Savings : ℝ := MarcusMaxPrice - FinalPrice

theorem marcus_savings : Savings = 46 := by
  sorry

end marcus_savings_l1514_151403


namespace august_first_problem_answer_l1514_151428

theorem august_first_problem_answer (A : ℕ)
  (h1 : 2 * A = B)
  (h2 : 3 * A - 400 = C)
  (h3 : A + B + C = 3200) : A = 600 :=
sorry

end august_first_problem_answer_l1514_151428


namespace solve_for_b_l1514_151465

theorem solve_for_b (b : ℚ) (h : b + b / 4 - 1 = 3 / 2) : b = 2 :=
sorry

end solve_for_b_l1514_151465


namespace find_tan_alpha_plus_pi_div_12_l1514_151434

theorem find_tan_alpha_plus_pi_div_12 (α : ℝ) (h : Real.sin α = 3 * Real.sin (α + Real.pi / 6)) :
  Real.tan (α + Real.pi / 12) = 2 * Real.sqrt 3 - 4 :=
by
  sorry

end find_tan_alpha_plus_pi_div_12_l1514_151434


namespace infinite_series_sum_l1514_151487

theorem infinite_series_sum :
  (∑' n : Nat, (4 * n + 1) / ((4 * n - 1)^2 * (4 * n + 3)^2)) = 1 / 72 :=
by
  sorry

end infinite_series_sum_l1514_151487


namespace parallel_lines_k_l1514_151492

theorem parallel_lines_k (k : ℝ) 
  (h₁ : k ≠ 0)
  (h₂ : ∀ x y : ℝ, (x - k * y - k = 0) = (y = (1 / k) * x - 1))
  (h₃ : ∀ x : ℝ, (y = k * (x - 1))) :
  k = -1 :=
by
  sorry

end parallel_lines_k_l1514_151492


namespace bob_bakes_pie_in_6_minutes_l1514_151460

theorem bob_bakes_pie_in_6_minutes (x : ℕ) (h_alice : 60 / 5 = 12)
  (h_condition : 12 - 2 = 60 / x) : x = 6 :=
sorry

end bob_bakes_pie_in_6_minutes_l1514_151460


namespace largest_3_digit_sum_l1514_151457

-- Defining the condition that ensures X, Y, Z are different digits ranging from 0 to 9
def valid_digits (X Y Z : ℕ) : Prop :=
  X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧ X < 10 ∧ Y < 10 ∧ Z < 10

-- Problem statement: Proving the largest possible 3-digit sum is 994
theorem largest_3_digit_sum : ∃ (X Y Z : ℕ), valid_digits X Y Z ∧ 111 * X + 11 * Y + Z = 994 :=
by
  sorry

end largest_3_digit_sum_l1514_151457


namespace no_real_roots_l1514_151408

noncomputable def polynomial (p : ℝ) (x : ℝ) : ℝ :=
  x^4 + 4 * p * x^3 + 6 * x^2 + 4 * p * x + 1

theorem no_real_roots (p : ℝ) :
  (p > -Real.sqrt 5 / 2) ∧ (p < Real.sqrt 5 / 2) ↔ ¬(∃ x : ℝ, polynomial p x = 0) := by
  sorry

end no_real_roots_l1514_151408


namespace a4_is_5_l1514_151482

-- Definitions based on the given conditions in the problem
def sum_arith_seq (n a1 d : ℤ) : ℤ := n * a1 + (n * (n-1)) / 2 * d

def S6 : ℤ := 24
def S9 : ℤ := 63

-- The proof problem: we need to prove that a4 = 5 given the conditions
theorem a4_is_5 (a1 d : ℤ) (h_S6 : sum_arith_seq 6 a1 d = S6) (h_S9 : sum_arith_seq 9 a1 d = S9) : 
  a1 + 3 * d = 5 :=
sorry

end a4_is_5_l1514_151482


namespace probability_no_intersecting_chords_l1514_151474

open Nat

def double_factorial (n : Nat) : Nat :=
  if n = 0 ∨ n = 1 then 1 else n * double_factorial (n - 2)

def catalan_number (n : Nat) : Nat :=
  (factorial (2 * n)) / (factorial n * factorial (n + 1))

theorem probability_no_intersecting_chords (n : Nat) (h : n > 0) :
  (catalan_number n) / (double_factorial (2 * n - 1)) = 2^n / (factorial (n + 1)) :=
by
  sorry

end probability_no_intersecting_chords_l1514_151474


namespace sqrt_720_simplified_l1514_151431

theorem sqrt_720_simplified : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  have h1 : 720 = 2^4 * 3^2 * 5 := by norm_num
  -- Here we use another proven fact or logic per original conditions and definition
  sorry

end sqrt_720_simplified_l1514_151431


namespace bank_exceeds_50_dollars_l1514_151419

theorem bank_exceeds_50_dollars (a : ℕ := 5) (r : ℕ := 2) :
  ∃ n : ℕ, 5 * (2 ^ n - 1) > 5000 ∧ (n ≡ 9 [MOD 7]) :=
by
  sorry

end bank_exceeds_50_dollars_l1514_151419


namespace functions_equal_l1514_151495

noncomputable def f (x : ℝ) : ℝ := x^0
noncomputable def g (x : ℝ) : ℝ := x / x

theorem functions_equal (x : ℝ) (hx : x ≠ 0) : f x = g x :=
by
  unfold f g
  sorry

end functions_equal_l1514_151495


namespace tan_alpha_eq_one_l1514_151493

open Real

theorem tan_alpha_eq_one (α : ℝ) (h : (sin α + cos α) / (2 * sin α - cos α) = 2) : tan α = 1 := 
by
  sorry

end tan_alpha_eq_one_l1514_151493


namespace min_reciprocal_sum_l1514_151481

theorem min_reciprocal_sum (m n a b : ℝ) (h1 : m = 5) (h2 : n = 5) 
  (h3 : m * a + n * b = 1) (h4 : 0 < a) (h5 : 0 < b) : 
  (1 / a + 1 / b) = 20 :=
by 
  sorry

end min_reciprocal_sum_l1514_151481


namespace intersection_of_A_and_B_l1514_151430

noncomputable def A : Set ℝ := {x | x^2 - 1 ≤ 0}

noncomputable def B : Set ℝ := {x | (x - 2) / x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l1514_151430


namespace boat_speed_greater_than_stream_l1514_151436

def boat_stream_speed_difference (S U V : ℝ) := 
  (S / (U - V)) - (S / (U + V)) + (S / (2 * V + 1)) = 1

theorem boat_speed_greater_than_stream 
  (S : ℝ) (U V : ℝ) 
  (h_dist : S = 1) 
  (h_time_diff : boat_stream_speed_difference S U V) :
  U - V = 1 :=
sorry

end boat_speed_greater_than_stream_l1514_151436


namespace area_N1N2N3_relative_l1514_151496

-- Definitions
variable (A B C D E F N1 N2 N3 : Type)
-- Assuming D, E, F are points on sides BC, CA, AB respectively such that CD, AE, BF are one-fourth of their respective sides.
variable (area_ABC : ℝ)  -- Total area of triangle ABC
variable (area_N1N2N3 : ℝ)  -- Area of triangle N1N2N3

-- Given conditions
variable (H1 : CD = 1 / 4 * BC)
variable (H2 : AE = 1 / 4 * CA)
variable (H3 : BF = 1 / 4 * AB)

-- The expected result
theorem area_N1N2N3_relative :
  area_N1N2N3 = 7 / 15 * area_ABC :=
sorry

end area_N1N2N3_relative_l1514_151496


namespace part_I_part_II_l1514_151458

def setA (x : ℝ) : Prop := 0 ≤ x - 1 ∧ x - 1 ≤ 2

def setB (x : ℝ) (a : ℝ) : Prop := 1 < x - a ∧ x - a < 2 * a + 3

def complement_R (x : ℝ) (a : ℝ) : Prop := x ≤ 2 ∨ x ≥ 6

theorem part_I (a : ℝ) (x : ℝ) (ha : a = 1) : 
  setA x ∨ setB x a ↔ (1 ≤ x ∧ x < 6) ∧ 
  (setA x ∧ complement_R x a ↔ 1 ≤ x ∧ x ≤ 2) := 
by
  sorry

theorem part_II (a : ℝ) : 
  (∃ x, setA x ∧ setB x a) ↔ -2/3 < a ∧ a < 2 := 
by
  sorry

end part_I_part_II_l1514_151458


namespace solve_for_x_l1514_151414

def δ (x : ℝ) : ℝ := 4 * x + 5
def φ (x : ℝ) : ℝ := 5 * x + 4

theorem solve_for_x (x : ℝ) (h : δ (φ x) = 4) : x = -17 / 20 := by
  sorry

end solve_for_x_l1514_151414


namespace problem_I_II_l1514_151433

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (2 * x + φ)

theorem problem_I_II
  (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi / 2)
  (h3 : f 0 φ = 1 / 2) :
  ∃ T : ℝ, T = Real.pi ∧ φ = Real.pi / 6 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → (f x φ) ≥ -1 / 2) :=
by
  sorry

end problem_I_II_l1514_151433


namespace trigonometry_identity_l1514_151442

theorem trigonometry_identity (α : ℝ) (P : ℝ × ℝ) (h : P = (4, -3)) :
  let x := P.1
  let y := P.2
  let r := Real.sqrt (x^2 + y^2)
  x = 4 →
  y = -3 →
  r = 5 →
  Real.tan α = y / x := by
  intros x y r hx hy hr
  rw [hx, hy]
  simp [Real.tan, div_eq_mul_inv, mul_comm]
  sorry

end trigonometry_identity_l1514_151442


namespace cost_of_eraser_pencil_l1514_151441

-- Define the cost of regular and short pencils
def cost_regular_pencil : ℝ := 0.5
def cost_short_pencil : ℝ := 0.4

-- Define the quantities sold
def quantity_eraser_pencils : ℕ := 200
def quantity_regular_pencils : ℕ := 40
def quantity_short_pencils : ℕ := 35

-- Define the total revenue
def total_revenue : ℝ := 194

-- Problem statement: Prove that the cost of a pencil with an eraser is 0.8
theorem cost_of_eraser_pencil (P : ℝ)
  (h : 200 * P + 40 * cost_regular_pencil + 35 * cost_short_pencil = total_revenue) :
  P = 0.8 := by
  sorry

end cost_of_eraser_pencil_l1514_151441


namespace lines_per_stanza_l1514_151400

-- Define the number of stanzas
def num_stanzas : ℕ := 20

-- Define the number of words per line
def words_per_line : ℕ := 8

-- Define the total number of words in the poem
def total_words : ℕ := 1600

-- Theorem statement to prove the number of lines per stanza
theorem lines_per_stanza : 
  (total_words / words_per_line) / num_stanzas = 10 := 
by sorry

end lines_per_stanza_l1514_151400
